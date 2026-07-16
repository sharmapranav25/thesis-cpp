"""Decoder-only pretraining for PARSeq (DPTR-style).

Trains PARSeq's decoder + token embeddings + head on next-token prediction
over text-only sequences (here: the 100k synth labels). Encoder is bypassed
by feeding zero memory to cross-attention, so only self-attention + FFN +
embeddings + head learn during this phase.

Result is saved as a state_dict that the full image-text trainer can load
as a warm-start. Useful when paired image-text data is limited; for uniform
random alphanumeric captchas the gain is expected to be small (no language
structure to learn beyond uniform priors)."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

PARSEQ_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "parseq"
sys.path.insert(0, str(PARSEQ_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

from src.parseq.train import build_parseq, CHARSET


class LabelsDataset(Dataset):
    """Reads filename,label CSV and returns label strings only."""
    def __init__(self, labels_csv: Path):
        with labels_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            self.labels = [row[1] for row in reader if row]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.labels[i]


def collate(batch):
    return list(batch)


def main(cfg: dict, args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_parseq(cfg).to(device)

    # Get tokenizer + dimensions from the lightning module
    tokenizer = model.tokenizer

    ds = LabelsDataset(Path(args.labels_csv))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=2, collate_fn=collate, drop_last=True)

    # Encoder forward determines memory shape; we bypass with zeros.
    # Compute the patch count from a dummy image forward (one-time).
    img_h, img_w = cfg["img_size"]
    dummy = torch.zeros(1, 3, img_h, img_w, device=device)
    with torch.inference_mode():
        mem = model.model.encode(dummy)
    n_patches, embed_dim = mem.shape[1], mem.shape[2]
    print(f"encoder yields ({n_patches} patches, dim={embed_dim}); using zero memory of that shape")

    # Optimizer over decoder + embeddings + head only
    parseq_model = model.model
    pretrain_params = list(parseq_model.decoder.parameters()) \
        + list(parseq_model.text_embed.parameters()) \
        + [parseq_model.pos_queries] \
        + list(parseq_model.head.parameters())
    optim = torch.optim.AdamW(pretrain_params, lr=args.lr, weight_decay=0.0)

    # Freeze encoder so it doesn't update during decoder pretraining
    for p in parseq_model.encoder.parameters():
        p.requires_grad = False

    pad_id = tokenizer.pad_id
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id

    parseq_model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"pretrain epoch {epoch+1}/{args.epochs}", ncols=120)
        running = 0.0
        n_seen = 0
        for batch in pbar:
            tokens = tokenizer.encode(batch, device)  # (B, L) — [BOS, c1..cN, EOS, PAD..]
            B, L = tokens.shape
            # Forward permutation only (canonical L-R)
            tgt_in = tokens[:, :-1]    # input: [BOS, c1..cN, EOS]
            tgt_out = tokens[:, 1:]    # target: [c1..cN, EOS, PAD..]
            tgt_mask = torch.triu(torch.ones(L - 1, L - 1, dtype=torch.bool, device=device), 1)
            tgt_padding_mask = (tgt_in == pad_id) | (tgt_in == eos_id)

            zero_memory = torch.zeros(B, n_patches, embed_dim, device=device)

            out = parseq_model.decode(
                tgt_in, zero_memory,
                tgt_mask=tgt_mask,
                tgt_padding_mask=tgt_padding_mask,
            )
            logits = parseq_model.head(out)  # (B, L-1, vocab)

            # Mask out PAD positions in loss
            loss = F.cross_entropy(
                logits.flatten(0, 1), tgt_out.flatten(),
                ignore_index=pad_id,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrain_params, max_norm=5.0)
            optim.step()

            running += loss.item() * B
            n_seen += B
            pbar.set_postfix({"loss": f"{running / n_seen:.4f}"})

    # Save full model state_dict so the image-text trainer can load it directly
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, out_path)
    print(f"saved pretrained decoder weights to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/parseq.yaml"))
    ap.add_argument("--labels-csv", type=Path, default=Path("data/synth_nobg/labels.csv"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/parseq_pretrained_decoder.pt"))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    main(cfg, args)
