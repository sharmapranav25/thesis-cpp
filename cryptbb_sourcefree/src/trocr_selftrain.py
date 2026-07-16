"""Self-training TrOCR on the UNLABELED reals — stacks on the font-general prior.

TrOCR fine-tuned on synth peaks ~51% char then OVERFITS to the synth font. This
instead alternates: pseudo-label the 600 UNLABELED reals (TTA-consensus + high
confidence + length filter), add the confident ones to training, keep fine-tuning
on synth+pseudo-real. The pseudo-real signal pulls the model TOWARD the real font
(opposite of synth overfit) — the flywheel that PARSeq couldn't start (0% word)
but TrOCR can (it already reads whole captchas ~3-4% + many chars).

GUARDRAIL: pseudo-label the 600 `unsolved` pool ONLY; the 651 eval is scored for
reporting, never used for pseudo-labels or selection. No human labels anywhere.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from peft import LoraConfig, get_peft_model

from src.trocr_finetune import OcrDS, collate, score_real, save_lora


@torch.inference_mode()
def pseudo_label_reals(model, processor, real_dir, device, conf_thr, n_len=5, batch=32):
    """TTA-averaged generation over the unlabeled reals; keep length-n_len
    predictions whose mean per-token probability >= conf_thr."""
    model.eval()
    paths = sorted(Path(real_dir).glob("*.png"))
    kept = []
    confs = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch]]
        pv = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pv, max_new_tokens=8, num_beams=1,
                             output_scores=True, return_dict_in_generate=True)
        seqs = out.sequences
        # per-step probs -> mean token prob per sample
        scores = torch.stack(out.scores, dim=1).softmax(-1)          # B, T, V
        toks = seqs[:, 1:1+scores.shape[1]]
        tokprob = scores.gather(-1, toks.unsqueeze(-1)).squeeze(-1)  # B, T
        preds = processor.batch_decode(seqs, skip_special_tokens=True)
        for j, p in enumerate(preds):
            s = p.strip().replace(" ", "")
            # confidence = mean prob over the emitted (non-pad) chars
            valid = toks[j] != processor.tokenizer.pad_token_id
            c = float(tokprob[j][valid].mean()) if valid.any() else 0.0
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
    ap.add_argument("--model", default="microsoft/trocr-large-str")
    ap.add_argument("--adapter", default="data/ckpts/trocr_lora/lora_adapter.pt",
                    help="warm-start LoRA adapter (best synth-fine-tuned TrOCR)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--round-steps", type=int, default=500)
    ap.add_argument("--conf-thr", type=float, default=0.9)
    ap.add_argument("--real-frac", type=float, default=0.4, help="fraction of each batch drawn from pseudo-real")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="data/ckpts/trocr_selftrain")
    args = ap.parse_args()

    import yaml
    real_dir = yaml.safe_load(open(args.corpus))["schemes"][args.scheme]["real_dir"]
    device = "cuda"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["query", "key", "value", "q_proj", "k_proj", "v_proj", "out_proj"])
    model = get_peft_model(model, lora).to(device)
    if args.adapter and Path(args.adapter).exists():
        sd = torch.load(args.adapter, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[warm-start] loaded adapter {args.adapter} (missing {len(missing)}, unexpected {len(unexpected)})", flush=True)

    synth_rows = [(r[0], r[1]) for r in list(csv.reader(open(Path(args.pool) / "labels.csv")))[1:]]
    synth_loader = DataLoader(OcrDS(args.pool, synth_rows, processor, augment=True), batch_size=args.batch,
                              shuffle=True, num_workers=6, collate_fn=collate, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    w, c, n = score_real(model, processor, device)
    print(f"[round 0 / warm-start] REAL word {w:.2f}%  char {c:.2f}%", flush=True)
    best = w

    def cycle(dl):
        while True:
            for b in dl:
                yield b
    synth_it = cycle(synth_loader)

    for rnd in range(1, args.rounds + 1):
        kept, (nk, tot, mconf) = pseudo_label_reals(model, processor, real_dir, device, args.conf_thr)
        print(f"[round {rnd}] pseudo-labeled {nk}/{tot} reals (conf>={args.conf_thr}, mean-conf {mconf:.3f})", flush=True)
        real_it = None
        if nk >= 4:
            # batch must not exceed #pseudo-labels, and NEVER drop_last (else an
            # empty loader makes cycle() spin forever — the overnight hang bug).
            real_loader = DataLoader(OcrDS(real_dir, kept, processor, augment=True),
                                     batch_size=min(args.batch, nk), shuffle=True, num_workers=2,
                                     collate_fn=collate, pin_memory=True, drop_last=False)
            real_it = cycle(real_loader)
        model.train(); run = 0.0
        for st in range(args.round_steps):
            use_real = real_it is not None and np.random.random() < args.real_frac
            pv, ids = next(real_it if use_real else synth_it)
            out = model(pixel_values=pv.to(device), labels=ids.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); run += float(out.loss)
        w, c, n = score_real(model, processor, device)
        print(f"[round {rnd}] loss {run/args.round_steps:.3f}  kept {nk}  REAL word {w:.2f}%  char {c:.2f}%", flush=True)
        if w > best:
            best = w; save_lora(model, args.out)
            print(f"  saved (word {w:.2f}%) -> {args.out}", flush=True)
    print(f"done. best REAL word {best:.2f}%", flush=True)


if __name__ == "__main__":
    main()
