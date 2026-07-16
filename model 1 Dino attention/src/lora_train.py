"""LoRA-tune the SD2.0 U-Net inside GlyphControl on real CryptBB captchas.

Why this works:
- The ControlNet branch already preserves glyph identity — we don't touch it.
- The U-Net only needs to learn the captcha *style* (speckle bg, vivid hues,
  chromatic shadows). Style fine-tuning is a marginal-distribution problem —
  we feed the U-Net real captchas and let it match their pixel statistics.
- We attach LoRA adapters to the attention projections in the U-Net, freeze
  everything else, and run standard latent-diffusion noise-prediction loss.

After training, `lora.pt` can be loaded back into the GlyphControl model at
inference time (see `apply_lora_state_dict` below) and combined with the
existing glyph control branch to produce label-faithful captcha-styled images.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm


REPO = Path(__file__).resolve().parent.parent
GLYPH_ROOT = REPO / "third_party" / "GlyphControl"
sys.path.insert(0, str(GLYPH_ROOT))


# ---------- LoRA primitives ----------------------------------------------

class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a low-rank trainable update.
    Name kept short so saved state-dict keys are short."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def disable_gradient_checkpointing(module: nn.Module) -> int:
    """ldm's custom checkpoint() passes self.parameters() to autograd.grad,
    which fails when most params are frozen. Easier to just disable it.
    Costs more VRAM but we're at batch=1-2 anyway."""
    n = 0
    for m in module.modules():
        for attr in ("checkpoint", "use_checkpoint"):
            if hasattr(m, attr) and isinstance(getattr(m, attr), bool):
                setattr(m, attr, False)
                n += 1
    return n


def attach_lora_to_unet(unet: nn.Module, rank: int = 8, alpha: float = 16.0) -> list[nn.Parameter]:
    """Wrap every CrossAttention to_q/to_k/to_v/to_out[0] with LoRALinear.
    Returns the list of newly-trainable LoRA params."""
    targets = ("to_q", "to_k", "to_v")
    params: list[nn.Parameter] = []
    n_wrapped = 0
    for parent_name, parent in unet.named_modules():
        # ldm CrossAttention has direct attributes to_q / to_k / to_v / to_out (Sequential).
        if not (hasattr(parent, "to_q") and hasattr(parent, "to_k") and hasattr(parent, "to_v")):
            continue
        for attr in targets:
            base = getattr(parent, attr)
            if isinstance(base, nn.Linear):
                wrapped = LoRALinear(base, rank=rank, alpha=alpha).to(base.weight.device, dtype=base.weight.dtype)
                setattr(parent, attr, wrapped)
                params += [wrapped.lora_A.weight, wrapped.lora_B.weight]
                n_wrapped += 1
        # to_out is a Sequential([Linear, Dropout]) in ldm; wrap the Linear inside.
        to_out = getattr(parent, "to_out", None)
        if isinstance(to_out, nn.Sequential) and isinstance(to_out[0], nn.Linear):
            base = to_out[0]
            wrapped = LoRALinear(base, rank=rank, alpha=alpha).to(base.weight.device, dtype=base.weight.dtype)
            to_out[0] = wrapped
            params += [wrapped.lora_A.weight, wrapped.lora_B.weight]
            n_wrapped += 1
    print(f"attached LoRA to {n_wrapped} attention linears")
    return params


def lora_state_dict(unet: nn.Module) -> dict:
    sd = {}
    for name, mod in unet.named_modules():
        if isinstance(mod, LoRALinear):
            sd[f"{name}.lora_A.weight"] = mod.lora_A.weight.detach().cpu()
            sd[f"{name}.lora_B.weight"] = mod.lora_B.weight.detach().cpu()
    return sd


def apply_lora_state_dict(unet: nn.Module, sd: dict) -> int:
    n = 0
    for name, mod in unet.named_modules():
        if isinstance(mod, LoRALinear):
            a_key = f"{name}.lora_A.weight"
            b_key = f"{name}.lora_B.weight"
            if a_key in sd and b_key in sd:
                mod.lora_A.weight.data.copy_(sd[a_key].to(mod.lora_A.weight.device, dtype=mod.lora_A.weight.dtype))
                mod.lora_B.weight.data.copy_(sd[b_key].to(mod.lora_B.weight.device, dtype=mod.lora_B.weight.dtype))
                n += 1
    return n


# ---------- Data ---------------------------------------------------------

class CaptchaImageFolder(Dataset):
    """Real captchas resized to 512x512 with letterbox (preserves aspect)."""

    def __init__(self, root: Path, size: int = 512):
        self.paths = sorted(p for p in root.iterdir()
                            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                            and not p.name.startswith("_"))
        self.size = size
        # Light augmentation: random horizontal flip would corrupt glyph orientation,
        # but we're training U-Net for *style*, not glyph identity (control branch
        # handles that). Still, captchas have an inherent left-to-right reading
        # direction — keep flip off. Just slight brightness/contrast jitter.
        self.tf = T.Compose([
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def _letterbox(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        canvas.paste(resized, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self._letterbox(img)
        return self.tf(img)


# ---------- Training -----------------------------------------------------

def unet_forward_train(unet, x, timesteps, context):
    """Mirror of ControlledUnetModel.forward but WITHOUT the no_grad wrap on
    input/middle blocks, and assuming control=None. Needed so LoRA in those
    blocks actually receives gradients."""
    from ldm.modules.diffusionmodules.util import timestep_embedding
    t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False)
    emb = unet.time_embed(t_emb)
    h = x.type(unet.dtype)
    hs = []
    for module in unet.input_blocks:
        h = module(h, emb, context)
        hs.append(h)
    h = unet.middle_block(h, emb, context)
    for module in unet.output_blocks:
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb, context)
    h = h.type(x.dtype)
    return unet.out(h)


def main(args):
    from cldm.hack import disable_verbosity
    from scripts.rendertext_tool import load_model_from_config
    from omegaconf import OmegaConf

    disable_verbosity()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("loading GlyphControl model...")
    cfg = OmegaConf.load(args.cfg)
    model = load_model_from_config(cfg, str(args.ckpt), verbose=False)
    model = model.cuda().eval()

    unet = model.model.diffusion_model  # ldm UNetModel
    vae = model.first_stage_model
    cond_stage = model.cond_stage_model  # frozen CLIP text encoder

    # Freeze everything; LoRA wrappers will register their params as trainable.
    for p in model.parameters():
        p.requires_grad_(False)

    n_off = disable_gradient_checkpointing(unet)
    print(f"disabled gradient checkpointing on {n_off} modules")

    lora_params = attach_lora_to_unet(unet, rank=args.rank, alpha=args.alpha)
    n_trainable = sum(p.numel() for p in lora_params)
    print(f"trainable LoRA params: {n_trainable / 1e6:.2f} M")

    # Tokenize the constant prompt once; FrozenOpenCLIPEmbedder returns shape
    # [B, 77, D]. We replicate to batch size at the call site.
    with torch.no_grad():
        c_text = model.get_learned_conditioning([args.prompt])  # [1, 77, D]
    c_text_dim = c_text.shape[-1]
    print(f"text cond shape: {c_text.shape}, dtype {c_text.dtype}")

    # Data
    ds = CaptchaImageFolder(args.data_dir)
    print(f"dataset: {len(ds)} real captchas at {args.data_dir}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    optim = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()

    # Linear warmup + cosine decay
    total = args.steps
    warmup = max(1, total // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    args.out.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=total, desc="lora")
    step = 0
    loader_it = iter(loader)
    losses = []
    while step < total:
        try:
            x = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            x = next(loader_it)
        x = x.cuda(non_blocking=True)
        b = x.shape[0]
        with torch.no_grad():
            # encode to latent, scale per LDM convention
            posterior = vae.encode(x)
            z0 = posterior.sample() * model.scale_factor  # [B, 4, 64, 64]

        t = torch.randint(0, model.num_timesteps, (b,), device=z0.device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = model.q_sample(x_start=z0, t=t, noise=noise)

        # We bypass ControlLDM.apply_model (which expects a control input) and
        # call the underlying diffusion_model directly with no ControlNet
        # residuals — equivalent to "style-only" training with an empty hint.
        c_ctx = c_text.expand(b, -1, -1)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            eps_pred = unet_forward_train(unet, zt, t, c_ctx)
            loss = F.mse_loss(eps_pred.float(), noise.float())

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        scaler.step(optim)
        scaler.update()
        sched.step()

        losses.append(loss.item())
        if step % 20 == 0:
            recent = np.mean(losses[-50:])
            pbar.set_postfix(loss=f"{recent:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
        pbar.update(1)
        step += 1

        if step % args.save_every == 0 or step == total:
            ckpt_path = args.out / f"lora_step{step}.pt"
            torch.save({
                "state_dict": lora_state_dict(unet),
                "rank": args.rank,
                "alpha": args.alpha,
                "step": step,
                "loss": float(np.mean(losses[-100:])),
                "prompt": args.prompt,
            }, ckpt_path)
            tqdm.write(f"saved {ckpt_path}  (avg loss last 100: {np.mean(losses[-100:]):.4f})")

    pbar.close()
    print(f"final avg loss (last 200): {np.mean(losses[-200:]):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("E:/pranav/model 1/data/raw"))
    ap.add_argument("--out", type=Path, default=REPO / "checkpoints" / "lora")
    ap.add_argument("--ckpt", type=Path,
                    default=GLYPH_ROOT / "checkpoints" / "textcaps5K_epoch_40_model_ema_only.ckpt")
    ap.add_argument("--cfg", type=Path, default=GLYPH_ROOT / "configs" / "config.yaml")
    ap.add_argument("--prompt", type=str, default="a captcha image")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
