"""Fine-tune Qwen2-VL-2B-Instruct (a true generalizing VLM, bigger font prior than
TrOCR) on our SOURCE-FREE synth; score on the 651 real. The advisor's core idea:
a broad pretrained VLM should transfer synth->real better than a font-specific
recognizer. LoRA adapts to the captcha task while preserving the prior.

GUARDRAIL: train on synth only; the 651 eval is scored for reporting, never trained.
Downloads go to E:\\hf_cache (system HF_HOME). Single A4000 16GB: bf16 + LoRA +
gradient checkpointing + small image resolution + grad-accum.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\hf_cache")          # keep downloads on E drive

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

PROMPT = "What are the 5 characters in this captcha? Reply with only the 5 characters, no spaces."


def make_prompt(n_len, upper):
    kind = "uppercase letters and digits" if upper else "characters"
    return f"What are the {n_len} characters in this captcha? Reply with only the {n_len} {kind}, no spaces."


def light_aug():
    return A.Compose([
        A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
        A.Affine(rotate=(-2, 2), scale=(0.97, 1.03), border_mode=0, fill=255, p=0.3),
    ])


def build_messages(img, answer=None, prompt=PROMPT):
    user = {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
    msgs = [user]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


class QwenOcrDS(Dataset):
    def __init__(self, root, rows, processor, augment=True, max_side=196):
        self.root = Path(root); self.rows = rows; self.proc = processor
        self.aug = light_aug() if augment else None
        self.max_side = max_side

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fn, lbl = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.aug is not None:
            img = Image.fromarray(self.aug(image=np.array(img))["image"])
        # keep the captcha at a modest resolution to limit visual tokens
        img = img.resize((self.max_side, max(28, int(self.max_side * img.height / img.width))))
        return img, lbl


def make_collate(processor, prompt=PROMPT):
    from qwen_vl_utils import process_vision_info

    def collate(batch):
        texts, all_imgs = [], []
        prompt_lens = []
        for img, lbl in batch:
            msgs = build_messages(img, lbl, prompt)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            # prompt-only length (to mask everything before the answer)
            pmsgs = build_messages(img, None, prompt)
            ptext = processor.apply_chat_template(pmsgs, tokenize=False, add_generation_prompt=True)
            prompt_lens.append(ptext)
            imgs, _ = process_vision_info(msgs)
            all_imgs.append(imgs)
        flat_imgs = [im for sub in all_imgs for im in sub]
        enc = processor(text=texts, images=flat_imgs, return_tensors="pt", padding=True)
        # build labels: mask prompt tokens + padding
        labels = enc["input_ids"].clone()
        for b in range(len(batch)):
            penc = processor(text=[prompt_lens[b]], images=all_imgs[b], return_tensors="pt")
            plen = penc["input_ids"].shape[1]
            labels[b, :plen] = -100
        labels[enc["input_ids"] == processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc

    return collate


@torch.inference_mode()
def score_real(model, processor, device, batch=8, max_side=196,
               eval_dir=r"E:/pranav/model 1/data/eval", n_len=5, prompt=PROMPT, upper=False):
    from qwen_vl_utils import process_vision_info
    evd = Path(eval_dir)
    labels = {r[0]: r[1] for r in list(csv.reader(open(evd / "labels.csv")))[1:]}
    files = [f for f in labels if (evd / f).exists()]
    model.eval()
    ok = cc = ct = 0
    for i in range(0, len(files), batch):
        texts, imgs_all = [], []
        for f in files[i:i+batch]:
            img = Image.open(evd / f).convert("RGB")
            img = img.resize((max_side, max(28, int(max_side * img.height / img.width))))
            msgs = build_messages(img, None, prompt)
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            ii, _ = process_vision_info(msgs); imgs_all.append(ii)
        flat = [im for sub in imgs_all for im in sub]
        enc = processor(text=texts, images=flat, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=n_len + 3, do_sample=False)
        trimmed = [g[len(inp):] for inp, g in zip(enc.input_ids, gen)]
        outs = processor.batch_decode(trimmed, skip_special_tokens=True)
        for f, o in zip(files[i:i+batch], outs):
            o = o.strip().upper() if upper else o.strip()
            p = "".join(ch for ch in o if ch.isalnum())[:n_len]
            l = labels[f]; ok += int(p == l)
            for a, b in zip(p.ljust(len(l)), l):
                ct += 1; cc += int(a == b)
    model.train()
    return 100 * ok / len(files), 100 * cc / ct, len(files)


def save_lora(model, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(sd, Path(out) / "lora_adapter.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synth_pool/cryptbb_trocr")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-dir", default=r"E:/pranav/model 1/data/eval")
    ap.add_argument("--n-len", type=int, default=5)
    ap.add_argument("--upper", action="store_true", help="uppercase output parsing (e.g. DarkBay)")
    ap.add_argument("--out", default="data/ckpts/qwen_lora")
    args = ap.parse_args()
    prompt = make_prompt(args.n_len, args.upper)
    ev = dict(eval_dir=args.eval_dir, n_len=args.n_len, prompt=prompt, upper=args.upper)

    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=256*28*28, max_pixels=768*28*28)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = [(r[0], r[1]) for r in list(csv.reader(open(Path(args.pool) / "labels.csv")))[1:]]
    ds = QwenOcrDS(args.pool, rows, processor, augment=True)
    # num_workers=0: the collate closure isn't picklable for Windows spawn, and
    # training is GPU-bound anyway so main-process data prep is fine.
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                        collate_fn=make_collate(processor, prompt), pin_memory=False, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"=== zero-shot (pretrained Qwen2-VL) on {args.eval_dir} ===", flush=True)
    w, c, n = score_real(model, processor, device, **ev)
    print(f"zero-shot: REAL word {w:.2f}% char {c:.2f}%  (n={n})", flush=True)

    import time
    best = -1.0; step = 0; run = 0.0; t0 = time.time()
    model.train()
    while step < args.steps:
        for enc in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            (out.loss / args.accum).backward()
            run += float(out.loss)
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"  step {step} loss {run/50:.3f}  {step/(time.time()-t0):.2f} it/s", flush=True)
                run = 0.0
            if step % args.eval_every == 0:
                w, c, n = score_real(model, processor, device, **ev)
                print(f"[step {step}] REAL word {w:.2f}%  char {c:.2f}%", flush=True)
                if w > best:
                    best = w; save_lora(model, args.out)
                    print(f"  saved (word {w:.2f}%) -> {args.out}", flush=True)
            if step >= args.steps:
                break
    print(f"done. best REAL word {best:.2f}%", flush=True)


if __name__ == "__main__":
    main()
