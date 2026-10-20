"""CAPRI Decoder visual core: larger CNN + DINOv3 + character-instance queries.

Construction requires pretrained weights unless explicitly running architecture
diagnostics. Supporting font-attribute/appearance task contracts remain separate.
"""
import contextlib
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .attention import MemoryLayer, QueryLayer, spatial_positions
from .common import sha
from .appearance import ATTRIBUTE_DIMS


def mlp(inputs, outputs, hidden=256):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False),
                                  nn.GroupNorm(16, outputs), nn.GELU(),
                                  nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False), nn.GroupNorm(16, outputs))
        self.skip = (nn.Identity() if inputs == outputs and stride == 1 else
                     nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.GroupNorm(16, outputs)))

    def forward(self, x): return F.gelu(self.main(x)+self.skip(x))


class VisualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList()
        inputs = 3
        for width in (64, 128, 256, 512):
            self.stages.append(nn.Sequential(ResidualBlock(inputs, width, 2), ResidualBlock(width, width)))
            inputs = width

    def forward(self, x):
        maps = []
        for stage in self.stages:
            x = stage(x); maps.append(x)
        return maps


def xyxy(center_size):
    return torch.cat((center_size[..., :2]-center_size[..., 2:]*.5,
                      center_size[..., :2]+center_size[..., 2:]*.5), -1).clamp(0, 1)


def aspect_region_pool(feature, boxes, masks, valid):
    """16-square letterbox crop, predicted soft gates, detached localization.

The crop is square in IMAGE pixels, not normalized x/y units. Cropped padding
and the letterbox surrounding a nonsquare glyph contribute zero feature values.
"""
    b, c, h, w = feature.shape; q = boxes.shape[1]
    boxes = boxes.detach().float(); masks = masks.detach().float().sigmoid()
    lo, hi = boxes[..., :2], boxes[..., 2:]
    center = (lo+hi)*.5
    scale = boxes.new_tensor((w, h))
    side = ((hi-lo)*scale).amax(-1).clamp_min(1.)
    axis = (torch.arange(16, device=feature.device).float()+.5)/16-.5
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    grid = center[:, :, None, None]+torch.stack((xx, yy), -1)*side[:, :, None, None, None]/scale
    inside = ((grid >= lo[:, :, None, None]) & (grid <= hi[:, :, None, None])).all(-1)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        # Stack query grids along output height; do not copy feature maps Q times.
        values = F.grid_sample(feature.float(), (grid*2-1).reshape(b, q*16, 16, 2), align_corners=False)
        values = values.reshape(b, c, q, 16, 16).permute(0, 2, 1, 3, 4)
        coverage = F.grid_sample(valid[:, None].float(), (grid*2-1).reshape(b, q*16, 16, 2), align_corners=False)
        coverage = coverage.reshape(b, q, 16, 16)
        gate = F.grid_sample(masks.reshape(b*q, 1, h, w), (grid*2-1).reshape(b*q, 16, 16, 2), align_corners=False)
        gate = .1+.9*gate.reshape(b, q, 16, 16)
        return (values*(coverage*gate*inside)[:, :, None]).reshape(b*q, c, 16, 16).to(feature.dtype)


class CapriDecoder(nn.Module):
    def __init__(self, dino_source, checkpoint=None, diagnostic_random=False, backbone='dinov3_vits16'):
        super().__init__()
        if checkpoint is None and not diagnostic_random:
            raise ValueError('Pretrained DINO checkpoint required; random mode is diagnostic only')
        if checkpoint is not None and diagnostic_random:
            raise ValueError('Choose pretrained or diagnostic construction explicitly')
        source = Path(dino_source).resolve()
        if backbone not in ('dinov3_vits16','dinov2_vits14_reg'): raise ValueError('Unsupported visual backbone')
        package = 'dinov2' if backbone == 'dinov2_vits14_reg' else 'dinov3'
        if not (source/package/'hub/backbones.py').is_file(): raise FileNotFoundError(source)
        sys.path.insert(0, str(source)) if str(source) not in sys.path else None
        import importlib
        backbones = importlib.import_module(package+'.hub.backbones')
        if not Path(backbones.__file__).resolve().is_relative_to(source):
            raise ValueError('DINO import resolved outside configured source')
        self.backbone = backbone
        self.patch_size = 14 if package == 'dinov2' else 16
        self.dino = getattr(backbones,backbone)(pretrained=False)
        self.pretrained_sha256 = None
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            self.dino.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            self.pretrained_sha256 = sha(checkpoint)
        self.diagnostic_random = diagnostic_random
        self.cnn = VisualCNN()
        self.dino_projection = nn.Conv2d(384, 256, 1)
        self.laterals = nn.ModuleList([nn.Conv2d(width, 256, 1) for width in (128, 256, 512)])
        self.refine = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                                   nn.GroupNorm(16, 256), nn.GELU()) for _ in range(3)])
        self.scale_position = nn.Parameter(torch.randn(3, 256)*.02)
        self.memory = nn.ModuleList([MemoryLayer() for _ in range(2)])
        self.query_content = nn.Embedding(16, 256); self.query_position = nn.Embedding(16, 256)
        self.initial_reference = nn.Linear(256, 4)
        self.decoder = nn.ModuleList([QueryLayer() for _ in range(4)])
        self.box_heads = nn.ModuleList([mlp(256, 4) for _ in range(4)])
        for head in self.box_heads:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        self.presence = nn.Linear(256, 1); nn.init.constant_(self.presence.bias, -2.)
        self.visibility = nn.Linear(256, 1)
        self.mask_query = mlp(256, 256); self.mask_pixel = nn.Conv2d(256, 256, 1)
        self.count = mlp(513, 7)
        self.effect_presence = mlp(768, 37); self.effect_intensity = mlp(768, 37)
        self.composition = mlp(512, 3); self.rows = mlp(512, 2)
        self.character_colour = mlp(256, 3)
        self.background_under = nn.Conv2d(256, 3, 1)
        self.font_reduce = nn.Conv2d(256, 64, 1)
        self.font_spatial = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),
                                          nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(), nn.Flatten())
        self.font_project = mlp(64*4*4+256, 128)
        self.font_style = mlp(64*4*4+256, 128)
        self.font_attributes = nn.ModuleDict({key:nn.Linear(128,size) for key,size in ATTRIBUTE_DIMS.items()})
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.set_dino_trainable(0)

    def set_dino_trainable(self, last_blocks):
        if last_blocks not in (0, 4): raise ValueError('Supported schedules: frozen or last four DINO blocks')
        self.dino_last_blocks = last_blocks
        self.dino.requires_grad_(False)
        if last_blocks:
            for block in self.dino.blocks[-last_blocks:]: block.requires_grad_(True)
            self.dino.norm.requires_grad_(True)
        # Keep the pretrained backbone's stochastic behavior disabled in either phase.
        self.dino.eval()

    def train(self, mode=True):
        super().train(mode); self.dino.eval(); return self

    @staticmethod
    def pool(feature, valid):
        weight = valid[:, None].to(feature.dtype)
        return (feature*weight).sum((-2, -1))/weight.sum((-2, -1)).clamp_min(1)

    def forward(self, pixels, valid_pixels):
        if pixels.ndim != 4 or pixels.shape[1] != 3 or valid_pixels.shape != (pixels.shape[0], *pixels.shape[-2:]):
            raise ValueError('Expected batched RGB and pixel-derived validity mask')
        b, _, h, w = pixels.shape
        if h % 16 or w % 16 or h > 128 or w > 768 or min(h, w) < 16 or valid_pixels.dtype != torch.bool:
            raise ValueError('Expected deterministic multiple-of-16 geometry within 128x768')
        if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError('RGB must be finite in [0,1]')
        heights = valid_pixels.any(-1).sum(-1); widths = valid_pixels.any(-2).sum(-1)
        yy = torch.arange(h, device=pixels.device)[None, :, None]
        xx = torch.arange(w, device=pixels.device)[None, None, :]
        expected = (yy < heights[:, None, None]) & (xx < widths[:, None, None])
        if (heights < 1).any() or (widths < 1).any() or not torch.equal(expected, valid_pixels):
            raise ValueError('Validity must be a nonempty top-left image rectangle')
        if (h-heights >= 16).any() or (w-widths >= 16).any():
            raise ValueError('Batch-dependent extra padding is forbidden')
        # Canonicalize pad colour; no hidden content is allowed through padding.
        pixels = torch.where(valid_pixels[:, None], pixels, torch.ones_like(pixels))
        x = (pixels-self.rgb_mean)/self.rgb_std
        cnn = self.cnn(x)
        context = torch.no_grad() if self.dino_last_blocks == 0 else contextlib.nullcontext()
        # Independent right/bottom WHITE padding to the encoder's patch size.
        # Neither image content nor glyph aspect ratio is stretched for DINOv2.
        dh, dw = (-h) % self.patch_size, (-w) % self.patch_size
        dino_pixels = F.pad(pixels,(0,dw,0,dh),value=1.)
        with context:
            tokens = self.dino.forward_features((dino_pixels-self.rgb_mean)/self.rgb_std)['x_norm_patchtokens']
        dino_map = tokens.transpose(1, 2).reshape(b,384,(h+dh)//self.patch_size,(w+dw)//self.patch_size)
        if self.patch_size != 16:
            dino_map = align_patch_grid(dino_map,(h,w),self.patch_size)
        maps = [lateral(m) for lateral, m in zip(self.laterals, cnn[1:])]
        maps[2] = self.refine[2](maps[2]+self.dino_projection(dino_map))
        maps[1] = self.refine[1](maps[1]+F.interpolate(maps[2], size=maps[1].shape[-2:], mode='bilinear', align_corners=False))
        maps[0] = self.refine[0](maps[0]+F.interpolate(maps[1], size=maps[0].shape[-2:], mode='bilinear', align_corners=False))
        valid = [F.adaptive_avg_pool2d(valid_pixels[:, None].float(), m.shape[-2:])[:, 0] for m in maps]
        position = torch.cat([spatial_positions(*m.shape[-2:], 256, x.device)+self.scale_position[i]
                              for i, m in enumerate(maps)], 0)[None]
        for layer in self.memory: maps = layer(maps, valid, position)
        query = self.query_content.weight[None].expand(b, -1, -1)
        query_pos = self.query_position.weight[None].expand(b, -1, -1)
        reference = self.initial_reference(query_pos).float().sigmoid()
        mask_pixels = self.mask_pixel(maps[0]); predictions = []
        global_feature = self.pool(maps[2], valid[2])
        for layer, box_head in zip(self.decoder, self.box_heads):
            query = layer(query, query_pos, maps, valid, reference)
            refined = (torch.logit(reference.clamp(1e-4, 1-1e-4))+box_head(query).float()).sigmoid()
            presence = self.presence(query).squeeze(-1)
            weighted_query = (query*presence.sigmoid()[..., None]).sum(1)
            masks = torch.einsum('bqc,bchw->bqhw', self.mask_query(query), mask_pixels)/(256**.5)
            predictions.append(dict(presence_logits=presence, boxes=xyxy(refined), mask_logits=masks,
                                    count_logits=self.count(torch.cat((global_feature, weighted_query,
                                                                      presence.sigmoid().sum(1, keepdim=True)), -1))))
            reference = refined.detach()
        pred = predictions[-1]
        crops = aspect_region_pool(self.font_reduce(maps[0]), pred['boxes'], pred['mask_logits'], valid[0])
        font_features = torch.cat((self.font_spatial(crops).reshape(b, 16, -1), query), -1)
        fonts = self.font_project(font_features)
        style = self.font_style(font_features)
        appearance = torch.cat((global_feature, weighted_query), -1)
        effects = torch.cat([self.pool(m, v) for m, v in zip(maps, valid)], -1)
        pred.update(font_embeddings=F.normalize(fonts.float(), dim=-1), visibility_logits=self.visibility(query).squeeze(-1),
                    effect_presence_logits=self.effect_presence(effects), intensity_predictions=self.effect_intensity(effects).sigmoid(),
                    composition_logits=self.composition(appearance), row_logits=self.rows(appearance),
                    colour_rgb=self.character_colour(query).sigmoid(), background_under=self.background_under(maps[0]).sigmoid(),
                    font_attribute_logits={key:head(style) for key,head in self.font_attributes.items()},
                    auxiliary=predictions[:-1])
        return pred

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()), trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    dino=sum(p.numel() for p in self.dino.parameters()), cnn=sum(p.numel() for p in self.cnn.parameters()))

    def resume_configuration(self):
        return dict(architecture='capri-decoder-core-v3-patch-aligned', backbone=self.backbone,
                    patch_alignment='physical-centres-white-pad-v1', dino_last_blocks=self.dino_last_blocks,
                    diagnostic_random=self.diagnostic_random, pretrained_sha256=self.pretrained_sha256)


def align_patch_grid(feature, image_hw, patch_size):
    """Sample patch-centre features at the stride-16 fusion grid's cell centres."""
    h,w=image_hw;fh,fw=feature.shape[-2:]
    y=(torch.arange(h//16,device=feature.device,dtype=torch.float32)+.5)*16/(fh*patch_size)
    x=(torch.arange(w//16,device=feature.device,dtype=torch.float32)+.5)*16/(fw*patch_size)
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    grid=torch.stack((xx,yy),-1)[None].expand(feature.shape[0],-1,-1,-1)*2-1
    with torch.autocast(device_type=feature.device.type,enabled=False):
        return F.grid_sample(feature.float(),grid,align_corners=False,padding_mode='border').to(feature.dtype)
