"""LoRA fine-tune Qwen2-VL-2B on PHP-MyBB synth captchas.

Adapter is attached to the LM-side attention projections only
(q_proj/k_proj/v_proj/o_proj). Vision tower stays frozen — its job (read
glyph shapes) doesn't change between zero-shot and fine-tuned.

Usage:
    python -m src.qwenvl.train --steps 2000 --grad-accum 8 --lr 1e-4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from peft import LoraConfig, get_peft_model

from src.qwenvl.data import CaptchaDataset, collate_single


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synth_mybb")
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--out", default="checkpoints/qwenvl_lora")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.to("cuda")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train:,} / {n_total:,} ({n_train/n_total:.2%})", flush=True)

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ds = CaptchaDataset(args.data, processor, max_samples=args.max_samples)
    print(f"dataset: {len(ds)} samples", flush=True)
    loader = DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=2,
        collate_fn=collate_single, pin_memory=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        prog = min(max(prog, 0.0), 1.0)
        # cosine to 10% of peak lr
        import math
        return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    model.train()
    step = 0
    accum = 0
    loss_running = 0.0
    t0 = time.time()
    log_path = out_dir / "train.log"
    log_f = open(log_path, "a", buffering=1)

    print(f"training for {args.steps} optimizer steps "
          f"(effective batch {args.grad_accum})", flush=True)

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            loss_running += loss.item() * args.grad_accum
            accum += 1
            if accum == args.grad_accum:
                cur_lr = lr_at(step)
                for g in optim.param_groups:
                    g["lr"] = cur_lr
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                accum = 0
                if step % args.log_every == 0:
                    avg = loss_running / (args.log_every * args.grad_accum)
                    dt = time.time() - t0
                    msg = (f"step {step}/{args.steps}  loss {avg:.4f}  "
                           f"lr {cur_lr:.2e}  {dt/step:.2f}s/step")
                    print(msg, flush=True)
                    log_f.write(msg + "\n")
                    loss_running = 0.0
                if step % args.save_every == 0 or step == args.steps:
                    ckpt = out_dir / f"step_{step}"
                    model.save_pretrained(ckpt)
                    print(f"saved {ckpt}", flush=True)

    log_f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
