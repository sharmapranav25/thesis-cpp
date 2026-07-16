"""Frozen PARSeq reader — the legibility judge (L_rec) and, at eval, the scorer.

Wraps `model 1`'s PARSeq: loads a checkpoint, exposes
  .read(imgs)      -> list[str]        argmax decode (for the M0/M2 read-back check)
  .rec_loss(x, y)  -> scalar Tensor    differentiable CE (the recognition anchor)

The reader expects images normalized to [-1,1] at the PARSeq input size (H,W =
32,128). `rec_loss` runs a single canonical-permutation teacher-forced decode
(mirrors PARSeq.training_step's forward perm) so gradients flow back to the
image through the ViT encoder — that is the legibility gradient for the GAN.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# sqlite/native DLL shim (PLAN.md) — PARSeq's base.py imports nltk->sqlite3.
_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

_MODEL1 = Path(r"E:/pranav/model 1")
_PARSEQ_TP = _MODEL1 / "third_party" / "parseq"
for p in (str(_MODEL1), str(_PARSEQ_TP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
import torch.nn.functional as _F
import yaml

from src.parseq.train import build_parseq, CHARSET  # noqa: E402


# Default: model 1's parseq_v2 architecture + its 57.26 val ckpt (a general
# reader, itself trained label-free on synth, NEVER on the 651 eval set).
DEFAULT_CFG = _MODEL1 / "configs" / "parseq_v2.yaml"
DEFAULT_CKPT = _MODEL1 / "checkpoints" / "parseq_v2" / "parseq-epoch=11-val_accuracy=57.260.ckpt"


class ParseqReader:
    def __init__(
        self,
        ckpt: str | Path = DEFAULT_CKPT,
        cfg: str | Path = DEFAULT_CFG,
        device: str = "cuda",
        train: bool = False,
        max_label_length: int | None = 6,
        strict: bool = False,
        refine_iters: int | None = 0,
    ):
        self.cfg = yaml.safe_load(Path(cfg).read_text())
        if max_label_length is not None:
            self.cfg["max_label_length"] = max_label_length
        # PARSeq's refinement pass (refine_iters>0) uses a cloze mask that, for our
        # fine-tuned reader, corrupts the FIRST decoded position (only char 0 misreads;
        # 1..N are fine). refine_iters=0 fixes it: AR force-len jumps 5%->90%.
        if refine_iters is not None:
            self.cfg["refine_iters"] = refine_iters
        self.img_size = tuple(self.cfg["img_size"])  # (H, W) = (32,128)
        self.device = device
        model = build_parseq(self.cfg)
        state = torch.load(Path(ckpt), map_location=device, weights_only=False)
        sd = state["state_dict"] if "state_dict" in state else state
        if strict:
            model.load_state_dict(sd)
        else:
            own = model.state_dict()
            keep = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
            partial = []
            # For pos_queries (decoder positional query embeddings), a plain drop
            # re-initializes them from scratch and position 0 fails to recover
            # (empirically: only the first char misreads). Instead PARTIAL-COPY the
            # overlapping rows from the source ckpt so the learned per-position
            # queries (incl. position 0) initialize correctly.
            for k, v in sd.items():
                if k in own and v.shape != own[k].shape and "pos_queries" in k:
                    tgt = own[k].clone()
                    n = min(v.shape[1], tgt.shape[1])
                    tgt[:, :n] = v[:, :n]
                    keep[k] = tgt
                    partial.append(f"{k} {tuple(v.shape)}->{tuple(own[k].shape)}")
            dropped = [k for k in sd if k not in keep]
            model.load_state_dict(keep, strict=False)
            if dropped or partial:
                print(f"[reader] init non-strict: {len(keep)}/{len(sd)} loaded; "
                      f"partial-copied {partial}; dropped {dropped}")
        self.model = model.to(device)
        self.tok = self.model.tokenizer
        if not train:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    # --- preprocessing -----------------------------------------------------
    def prep(self, x: torch.Tensor) -> torch.Tensor:
        """Resize a batch of images (already in [-1,1], RGB, any HxW) to the
        reader's input size. Differentiable (bilinear)."""
        if x.shape[-2:] != self.img_size:
            x = _F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        return x

    # --- inference (non-diff) ---------------------------------------------
    @torch.inference_mode()
    def read(self, x: torch.Tensor, lengths: list[int] | None = None,
             force_len: int | None = None) -> list[str]:
        """x: (B,3,H,W) in [-1,1]. Returns decoded strings.

        Default = greedy with EOS. These PARSeq nets emit EOS eagerly, so for a
        reliable read pass the known length: `lengths` (per-sample) or `force_len`
        (uniform). Force decoding masks the special tokens and argmaxes the first
        L positions — matches how model 1 scored at 57%."""
        x = self.prep(x.to(self.device))
        logits = self.model(x)
        if lengths is None and force_len is None:
            preds, _ = self.tok.decode(logits.softmax(-1))
            return list(preds)
        probs = logits.softmax(-1).cpu()
        itos = list(self.tok._itos)
        specials = {self.tok.eos_id, self.tok.bos_id, self.tok.pad_id}
        out = []
        for i in range(probs.shape[0]):
            n = force_len if force_len is not None else lengths[i]
            n = min(n, probs.shape[1])
            s = []
            for t in range(n):
                row = probs[i, t].clone()
                for sp in specials:
                    if sp < row.shape[0]:
                        row[sp] = -1.0
                s.append(itos[int(row.argmax())])
            out.append("".join(s))
        return out

    # --- recognition loss (diff) ------------------------------------------
    def rec_loss(self, x: torch.Tensor, labels: list[str]) -> torch.Tensor:
        """Differentiable CE of reading `labels` out of images `x` ([-1,1]).
        Single canonical (left-to-right) permutation, teacher-forced."""
        x = self.prep(x.to(self.device))
        tgt = self.tok.encode(labels, self.device)          # (B, L+2) BOS..EOS
        memory = self.model.model.encode(x)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        pad_id = self.model.pad_id
        eos_id = self.model.eos_id
        tgt_padding_mask = (tgt_in == pad_id) | (tgt_in == eos_id)
        # canonical forward permutation attn masks
        L = tgt.shape[1]
        perm = torch.arange(L, device=self.device)
        tgt_mask, query_mask = self.model.generate_attn_masks(perm)
        out = self.model.model.decode(
            tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask
        )
        logits = self.model.model.head(out).flatten(end_dim=1)
        loss = F.cross_entropy(logits, tgt_out.flatten(), ignore_index=pad_id)
        return loss


if __name__ == "__main__":
    # Smoke: load reader, print vocab size.
    r = ParseqReader()
    print("reader loaded; img_size", r.img_size, "vocab", len(r.tok))
