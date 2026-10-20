"""Portable multi-scale deformable attention and pixel-coordinate utilities.

Uses the sampling equations of Deformable DETR (Zhu et al., 2020), checked
against fundamentalvision/Deformable-DETR's Apache-2.0 reference implementation:
https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops
This implementation uses ordinary PyTorch autograd; no compiled extension.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def sample_scales(values, locations, weights):
    """values: B,H,D,Y,X per scale; locations: B,Q,H,L,P,2 in [0,1]."""
    b, q, heads, levels, points, _ = locations.shape
    if levels != len(values) or weights.shape != locations.shape[:-1]:
        raise ValueError('Attention scale/weight shape mismatch')
    dtype = values[0].dtype
    # FP32 bilinear accumulation remains stable and works under CUDA AMP.
    with torch.autocast(device_type=locations.device.type, enabled=False):
        result = locations.new_zeros((b, heads, values[0].shape[2], q), dtype=torch.float32)
        for level, value in enumerate(values):
            if value.shape[:2] != (b, heads):
                raise ValueError('Attention value batch/head mismatch')
            grid = locations[:, :, :, level].permute(0, 2, 1, 3, 4).reshape(b*heads, q, points, 2)
            samples = F.grid_sample(value.float().flatten(0, 1), grid.float()*2-1,
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
            scale_weights = weights[:, :, :, level].permute(0, 2, 1, 3).reshape(b*heads, 1, q, points)
            result = result + (samples*scale_weights.float()).sum(-1).reshape(b, heads, -1, q)
        return result.flatten(1, 2).transpose(1, 2).to(dtype)


class DeformableAttention(nn.Module):
    def __init__(self, width=256, heads=8, levels=3, points=4):
        super().__init__()
        if width % heads or min(width, heads, levels, points) <= 0:
            raise ValueError('Invalid deformable attention dimensions')
        self.width, self.heads, self.levels, self.points = width, heads, levels, points
        self.offset = nn.Linear(width, heads*levels*points*2)
        self.weight = nn.Linear(width, heads*levels*points)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.offset.weight)
        theta = torch.arange(heads)*2*math.pi/heads
        direction = torch.stack((theta.cos(), theta.sin()), -1)
        direction = direction / direction.abs().amax(-1, keepdim=True)
        radial = direction[:, None, None, :]*torch.arange(1, points+1)[None, None, :, None]
        with torch.no_grad():
            self.offset.bias.copy_(radial.expand(heads, levels, points, 2).reshape(-1))
        nn.init.zeros_(self.weight.weight); nn.init.zeros_(self.weight.bias)
        for layer in (self.value, self.output):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, query, maps, valid, reference):
        b, q, c = query.shape
        if c != self.width or len(maps) != self.levels or len(valid) != self.levels:
            raise ValueError('Unexpected feature dimensions')
        if reference.shape[:2] != (b, q) or reference.shape[-1] not in (2, 4):
            raise ValueError('Expected normalized point or center/size box references')
        offsets = self.offset(query).reshape(b, q, self.heads, self.levels, self.points, 2)
        weights = self.weight(query).reshape(b, q, self.heads, self.levels*self.points)
        weights = weights.float().softmax(-1).reshape(b, q, self.heads, self.levels, self.points)
        values = []
        for feature, mask in zip(maps, valid):
            h, w = feature.shape[-2:]
            if feature.shape[:2] != (b, c) or mask.shape != (b, h, w):
                raise ValueError('Invalid feature coverage shape')
            value = self.value(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # Apply coverage after projection, so projection bias cannot fill padding.
            value = value * mask[:, None].to(value.dtype)
            values.append(value.reshape(b, self.heads, c//self.heads, h, w))
        center = reference[:, :, None, None, None, :2].float()
        if reference.shape[-1] == 2:
            sizes = query.new_tensor([[m.shape[-1], m.shape[-2]] for m in maps])
            locations = center + offsets.float()/sizes[None, None, None, :, None]
        else:
            locations = center + offsets.float()/self.points*reference[:, :, None, None, None, 2:].float()*.5
        return self.output(sample_scales(values, locations, weights))


def grid_points(h, w, device):
    y, x = torch.meshgrid((torch.arange(h, device=device)+.5)/h,
                          (torch.arange(w, device=device)+.5)/w, indexing='ij')
    return torch.stack((x, y), -1).reshape(-1, 2)


def spatial_positions(h, w, width, device):
    if width % 4: raise ValueError('Position width must be divisible by four')
    xy = grid_points(h, w, device)*2*math.pi
    scale = 10000**(torch.arange(width//4, device=device).float()/(width//4))
    phase = xy[:, :, None]/scale
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(1)


def split_maps(tokens, shapes):
    chunks = tokens.split([h*w for h, w in shapes], dim=1)
    return [t.transpose(1, 2).reshape(t.shape[0], t.shape[2], h, w) for t, (h, w) in zip(chunks, shapes)]


class MemoryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = DeformableAttention()
        self.norm1 = nn.LayerNorm(256); self.norm2 = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, maps, valid, position):
        shapes = [m.shape[-2:] for m in maps]
        tokens = torch.cat([m.flatten(2).transpose(1, 2) for m in maps], 1)
        reference = torch.cat([grid_points(h, w, tokens.device) for h, w in shapes], 0)
        reference = reference[None].expand(tokens.shape[0], -1, -1)
        hidden = self.norm1(tokens+self.attention(tokens+position, maps, valid, reference))
        hidden = self.norm2(hidden+self.ffn(hidden))
        return [m*(v>0)[:, None] for m, v in zip(split_maps(hidden, shapes), valid)]


class QueryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        self.cross_attention = DeformableAttention()
        self.norms = nn.ModuleList([nn.LayerNorm(256) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, query, position, maps, valid, reference):
        q = query+position
        query = self.norms[0](query+self.self_attention(q, q, query, need_weights=False)[0])
        query = self.norms[1](query+self.cross_attention(query+position, maps, valid, reference))
        return self.norms[2](query+self.ffn(query))
