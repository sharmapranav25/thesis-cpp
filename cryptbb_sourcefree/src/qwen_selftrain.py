"""Self-training Qwen2-VL on the UNLABELED reals — stacks on the fine-tuned VLM.

The synth-fine-tuned Qwen plateaus ~68% char / ~25% word. That means ~25% of the
600 UNLABELED reals are read EXACTLY right — a rich pool of correct pseudo-labels
(vs TrOCR's 10, PARSeq's 1). Alternating: pseudo-label the reals (confidence +
length filter), fine-tune on synth + confident pseudo-reals, repeat — pulls the
model toward the real font, breaking the synth ceiling. No human labels.

GUARDRAIL: pseudo-label the 600 `unsolved` pool ONLY; the 651 eval is scored for
reporting, never used for pseudo-labels or selection.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"E:\hf_cache")

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from src.qwen_finetune import (QwenOcrDS, make_collate, score_real, save_lora,
                               build_messages, PROMPT, make_prompt)


@torch.inference_mode()
def pseudo_label(model, processor, real_dir, device, conf_thr, max_side=196, batch=8,
                 n_len=5, prompt=PROMPT, upper=False):
    from qwen_vl_utils import process_vision_info
    model.eval()
    paths = sorted(Path(real_dir).glob("*.png"))
    kept, confs = [], []
    for i in range(0, len(paths), batch):
        texts, imgs_all = [], []
        for p in paths[i:i+batch]:
            img = Image.open(p).convert("RGB")
            img = img.resize((max_side, max(28, int(max_side * img.height / img.width))))
            msgs = build_messages(img, None, prompt)
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            ii, _ = process_vision_info(msgs); imgs_all.append(ii)
        flat = [im for sub in imgs_all for im in sub]
        enc = processor(text=texts, images=flat, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=8, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
        seqs = gen.sequences
        newlen = seqs.shape[1] - enc.input_ids.shape[1]
        scores = torch.stack(gen.scores, dim=1).softmax(-1)          # B, newT, V
        newtoks = seqs[:, enc.input_ids.shape[1]:]
        tokprob = scores.gather(-1, newtoks.unsqueeze(-1)).squeeze(-1)
        outs = processor.batch_decode(newtoks, skip_special_tokens=True)
        for j, o in enumerate(outs):
            o = o.strip().upper() if upper else o.strip()
            s = "".join(ch for ch in o if ch.isalnum())[:n_len]
            # confidence = mean prob over first n_len emitted tokens
            c = float(tokprob[j][:n_len].mean()) if tokprob.shape[1] >= 1 else 0.0
            confs.append(c)
            if len(s) == n_len and c >= conf_thr:
                kept.append((paths[i+j].name, s))
    model.train()
    return kept, (len(kept), len(paths), float(np.mean(confs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synth_pool/cryptbb_trocr")
    ap.add_argument("--corpus", default="configs/corpus.yaml")
    ap.add_argument("--scheme", default="cryptbb")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default="data/ckpts/qwen_lora/lora_adapter.pt")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--round-steps", type=int, default=400)
    ap.add_argument("--conf-thr", type=float, default=0.9)
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-dir", default=r"E:/pranav/model 1/data/eval")
    ap.add_argument("--n-len", type=int, default=5)
    ap.add_argument("--upper", action="store_true")
    ap.add_argument("--real-dir", default=None,
                    help="override unlabeled pool to pseudo-label (must be DISJOINT from --eval-dir)")
    ap.add_argument("--out", default="data/ckpts/qwen_selftrain")
    args = ap.parse_args()
    prompt = make_prompt(args.n_len, args.upper)
    ev = dict(eval_dir=args.eval_dir, n_len=args.n_len, prompt=prompt, upper=args.upper)

    device = "cuda"
    real_dir = args.real_dir or yaml.safe_load(open(args.corpus))["schemes"][args.scheme]["real_dir"]
    print(f"[self-train] pseudo-label pool: {real_dir}  |  test: {args.eval_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=256*28*28, max_pixels=768*28*28)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    if args.adapter and Path(args.adapter).exists():
        sd = torch.load(args.adapter, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"[warm-start] loaded {args.adapter}", flush=True)

    synth_rows = [(r[0], r[1]) for r in list(csv.reader(open(Path(args.pool) / "labels.csv")))[1:]]
    coll = make_collate(processor, prompt)
    synth_loader = DataLoader(QwenOcrDS(args.pool, synth_rows, processor, augment=True),
                              batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=coll, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def cycle(dl):
        while True:
            for b in dl:
                yield b
    synth_it = cycle(synth_loader)

    w, c, n = score_real(model, processor, device, **ev)
    print(f"[round 0 / warm-start] REAL word {w:.2f}%  char {c:.2f}%", flush=True)
    best = w; save_lora(model, args.out)

    for rnd in range(1, args.rounds + 1):
        kept, (nk, tot, mconf) = pseudo_label(model, processor, real_dir, device, args.conf_thr,
                                              n_len=args.n_len, prompt=prompt, upper=args.upper)
        print(f"[round {rnd}] pseudo-labeled {nk}/{tot} reals (conf>={args.conf_thr}, mean-conf {mconf:.3f})", flush=True)
        real_it = None
        if nk >= 4:
            real_loader = DataLoader(QwenOcrDS(real_dir, kept, processor, augment=True),
                                     batch_size=min(args.batch, nk), shuffle=True, num_workers=0,
                                     collate_fn=coll, drop_last=False)
            real_it = cycle(real_loader)
        model.train(); run = 0.0
        for st in range(args.round_steps):
            use_real = real_it is not None and np.random.random() < args.real_frac
            enc = next(real_it if use_real else synth_it)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc); (out.loss / args.accum).backward(); run += float(out.loss)
            if (st + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
        w, c, n = score_real(model, processor, device, **ev)
        print(f"[round {rnd}] loss {run/args.round_steps:.3f}  kept {nk}  REAL word {w:.2f}%  char {c:.2f}%", flush=True)
        if w > best:
            best = w; save_lora(model, args.out)
            print(f"  saved (word {w:.2f}%) -> {args.out}", flush=True)
    print(f"done. best REAL word {best:.2f}%", flush=True)


if __name__ == "__main__":
    main()
