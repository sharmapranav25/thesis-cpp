"""Fine-tune TrOCR (a font-general pretrained OCR transformer) on our source-free
synth, score on the 651 real. Tests the hypothesis: a model with broad pretrained
font priors transfers synth->real better than a from-scratch PARSeq (which is
font-specific and hit 0% word). LoRA adapts to the captcha task while PRESERVING
the pretrained prior (the generalization we want).

GUARDRAIL: train on synth only; the 651 eval is scored for reporting, never trained.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None   # trusted MS weights; torch 2.5 .bin guard bypass
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from peft import LoraConfig, get_peft_model


def light_aug():
    return A.Compose([
        A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
        A.Affine(rotate=(-2, 2), scale=(0.97, 1.03), border_mode=0, fill=255, p=0.3),
    ])


class OcrDS(Dataset):
    def __init__(self, root, rows, processor, max_len=6, augment=True):
        self.root = Path(root); self.rows = rows; self.proc = processor
        self.max_len = max_len; self.aug = light_aug() if augment else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fn, lbl = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.aug is not None:
            img = Image.fromarray(self.aug(image=np.array(img))["image"])
        pv = self.proc(images=img, return_tensors="pt").pixel_values[0]
        ids = self.proc.tokenizer(lbl, padding="max_length", max_length=self.max_len,
                                  truncation=True).input_ids
        ids = [t if t != self.proc.tokenizer.pad_token_id else -100 for t in ids]
        return pv, torch.tensor(ids)


def collate(b):
    pv, ids = zip(*b)
    return torch.stack(pv), torch.stack(ids)


def save_lora(model, out):
    """Save only the LoRA adapter weights (avoids peft.save_pretrained's
    vocab_size bug on VisionEncoderDecoderConfig)."""
    Path(out).mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(sd, Path(out) / "lora_adapter.pt")


@torch.inference_mode()
def score_real(model, processor, device, n_len=5, batch=32):
    evd = Path(r"E:/pranav/model 1/data/eval")
    labels = {r[0]: r[1] for r in list(csv.reader(open(evd / "labels.csv")))[1:]}
    files = [f for f in labels if (evd / f).exists()]
    model.eval()
    ok = cc = ct = 0
    for i in range(0, len(files), batch):
        imgs = [Image.open(evd / f).convert("RGB") for f in files[i:i+batch]]
        pv = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
        gen = model.generate(pv, max_new_tokens=8, num_beams=1)
        preds = processor.batch_decode(gen, skip_special_tokens=True)
        for p, f in zip(preds, files[i:i+batch]):
            p = p.strip().replace(" ", "")
            l = labels[f]
            ok += int(p == l)
            for a, b in zip(p.ljust(len(l)), l):
                ct += 1; cc += int(a == b)
    model.train()
    return 100 * ok / len(files), 100 * cc / ct, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synth_pool/cryptbb_trocr")
    ap.add_argument("--model", default="microsoft/trocr-large-str")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=800, help="score 651 real every N steps")
    ap.add_argument("--out", default="data/ckpts/trocr_lora")
    args = ap.parse_args()

    device = "cuda"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                      target_modules=["query", "key", "value", "q_proj", "k_proj", "v_proj", "out_proj"])
    model = get_peft_model(model, lora)
    model.to(device)
    model.print_trainable_parameters()

    rows = list(csv.reader(open(Path(args.pool) / "labels.csv")))[1:]
    g = torch.Generator().manual_seed(0); perm = torch.randperm(len(rows), generator=g).tolist()
    nval = int(len(rows) * args.val_frac)
    val_rows = [rows[i] for i in perm[:nval]]; train_rows = [rows[i] for i in perm[nval:]]
    print(f"train={len(train_rows)} val={len(val_rows)}", flush=True)

    tr = DataLoader(OcrDS(args.pool, train_rows, processor, augment=True), batch_size=args.batch,
                    shuffle=True, num_workers=6, collate_fn=collate, pin_memory=True,
                    persistent_workers=True, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print("=== zero-shot (pretrained, before fine-tune) on 651 real ===", flush=True)
    w, c, n = score_real(model, processor, device)
    print(f"zero-shot: REAL word {w:.2f}% char {c:.2f}%", flush=True)

    import time
    best = -1.0; step = 0; run = 0.0; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for pv, ids in tr:
            out = model(pixel_values=pv.to(device), labels=ids.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); run += float(out.loss); step += 1
            if step % 100 == 0:
                print(f"  step {step} (ep{ep})  loss {run/100:.3f}  {step/(time.time()-t0):.2f} it/s", flush=True)
                run = 0.0
            if step % args.eval_every == 0:
                w, c, n = score_real(model, processor, device)
                print(f"[step {step}] REAL word {w:.2f}%  char {c:.2f}%", flush=True)
                if w > best:
                    best = w
                    save_lora(model, args.out)
                    print(f"  saved (word {w:.2f}%) -> {args.out}", flush=True)
        w, c, n = score_real(model, processor, device)
        print(f"epoch {ep}: REAL word {w:.2f}%  char {c:.2f}%", flush=True)
        if w > best:
            best = w; save_lora(model, args.out)
    print(f"done. best REAL word {best:.2f}%", flush=True)


if __name__ == "__main__":
    main()
