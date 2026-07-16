"""PARSeq training for Model 2 — curriculum-driven captcha solver.

Reads (filename, label, difficulty, is_fixed_length) rows from labels.csv.
Difficulty in [0,1] is a per-image knob written by the PHP renderer.
is_fixed_length is a per-image binary flag (1 if rendered with --length-mode
fixed:N, 0 if rendered with --length-mode range:MIN-MAX).

Adds:
  - CurriculumSampler: per-epoch widens the d-window of training samples.
  - CurriculumCallback: updates sampler.d_max each epoch.
  - PARSeqWithLengthHead: PARSeq subclass that adds a binary classification
    head over mean-pooled encoder features, predicting is_fixed_length.
    Joint loss = L_char + lengthmode_weight * L_lengthmode.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Anaconda-base venv on Windows: the venv inherits Anaconda's stdlib but the
# native DLLs (sqlite3.dll, etc.) live under Library/bin and aren't on PATH by
# default. Add them so `import nltk` (used by PARSeq's base.py) doesn't blow
# up on its sqlite3 import chain.
_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

PARSEQ_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "parseq"
sys.path.insert(0, str(PARSEQ_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml

import albumentations as A

from strhub.models.parseq.system import PARSeq  # noqa: E402


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _to_rgb(im):
    return im.convert("RGB")


def _build_train_augment() -> A.Compose:
    """Diversity #1: heavy training-time augmentation. Each batch sees a
    different perturbation so the model can't memorise pixel patterns."""
    return A.Compose([
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08, p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.4),
        A.GaussNoise(p=0.3),
        A.ImageCompression(quality_range=(50, 95), p=0.4),
        A.Affine(
            rotate=(-3, 3),
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            scale=(0.95, 1.05),
            shear={"x": (-3, 3)},
            border_mode=0,
            fill=255,
            p=0.6,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(2, 8),
            hole_width_range=(2, 12),
            fill=255,
            p=0.3,
        ),
    ])


class CaptchaDir(Dataset):
    """Read labels.csv rows into a Dataset.

    Supported CSV column layouts:
      2-col: filename,label                                    (Model 1 legacy)
      4-col: filename,label,difficulty,is_fixed_length         (Model 2 base)
      5-col: filename,label,difficulty,is_fixed_length,bucket  (bucket-curriculum)

    Missing fields default to difficulty=1.0, is_fixed_length=1, bucket="hard".
    Returns (image, label, is_fixed_length) per __getitem__.
    self.buckets[i] is a string label per row (e.g. "easy", "medium", "hard")."""

    def __init__(
        self, root: Path, img_size: tuple[int, int],
        rows: list[tuple] | None = None, augment: bool = False,
    ):
        self.root = Path(root)
        self.img_size = img_size  # (H, W)
        if rows is None:
            rows = _read_rows(self.root)
        self.rows = rows
        # Materialize difficulty + is_fixed + bucket for sampler / loss.
        self.difficulties = np.array(
            [float(r[2]) for r in rows], dtype=np.float32,
        )
        self.is_fixed = np.array(
            [int(r[3]) for r in rows], dtype=np.float32,
        )
        self.buckets = np.array([r[4] for r in rows])
        self.aug = _build_train_augment() if augment else None
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fn, lbl, _d, _isfx, _bucket = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.aug is not None:
            arr = np.array(img)
            arr = self.aug(image=arr)["image"]
            img = Image.fromarray(arr)
        img = self.tf(img)
        return img, lbl, float(_isfx)


def captcha_collate(batch):
    imgs, labels, is_fixed = zip(*batch)
    return (
        torch.stack(imgs, 0),
        list(labels),
        torch.tensor(is_fixed, dtype=torch.float32),
    )


def captcha_collate_no_lengthmode(batch):
    """Drops is_fixed; returns (images, labels) for vanilla PARSeq training_step."""
    imgs, labels, _ = zip(*batch)
    return torch.stack(imgs, 0), list(labels)


def _read_rows(root: Path) -> list[tuple]:
    """Read labels.csv; pad to 5-tuple (filename,label,difficulty,is_fixed,bucket)."""
    with (root / "labels.csv").open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = []
        for r in reader:
            if len(r) == 2:
                rows.append((r[0], r[1], "1.0", "1", "hard"))
            elif len(r) == 4:
                rows.append((r[0], r[1], r[2], r[3], "hard"))
            elif len(r) == 5:
                rows.append((r[0], r[1], r[2], r[3], r[4]))
            else:
                raise ValueError(f"unexpected labels.csv columns: {r}")
    return rows


def split_rows(root: Path, val_frac: float, seed: int) -> tuple[list, list]:
    rows = _read_rows(root)
    g = torch.Generator().manual_seed(seed)
    n_val = int(len(rows) * val_frac)
    perm = torch.randperm(len(rows), generator=g).tolist()
    val_idx = set(perm[:n_val])
    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    return train_rows, val_rows


class CurriculumSampler(Sampler[int]):
    """Yields `length` indices each epoch, sampled uniformly with replacement
    from {i : difficulties[i] <= d_max}. d_max is mutable; CurriculumCallback
    updates it at each epoch start."""

    def __init__(self, difficulties: np.ndarray, length: int, d_max: float = 1.0):
        self.difficulties = difficulties
        self.length = length
        self.d_max = float(d_max)
        self._gen = np.random.default_rng()

    def __iter__(self):
        eligible = np.where(self.difficulties <= self.d_max)[0]
        if len(eligible) == 0:
            # Pathological: d_max so low nothing qualifies. Fall back to whole set.
            eligible = np.arange(len(self.difficulties))
        idx = self._gen.choice(eligible, size=self.length, replace=True)
        return iter(idx.tolist())

    def __len__(self):
        return self.length


class CurriculumCallback(pl.Callback):
    """Linear ramp: d_max = min(1, (epoch+1)/warmup_epochs)."""

    def __init__(self, sampler: CurriculumSampler, warmup_epochs: int):
        super().__init__()
        self.sampler = sampler
        self.warmup_epochs = max(1, int(warmup_epochs))

    def on_train_epoch_start(self, trainer, pl_module):
        e = trainer.current_epoch
        d_max = min(1.0, (e + 1) / self.warmup_epochs)
        self.sampler.d_max = d_max
        pl_module.log("curriculum_d_max", d_max, on_epoch=True, on_step=False)
        print(f"[curriculum] epoch={e}  d_max={d_max:.4f}")


class BucketCurriculumSampler(Sampler[int]):
    """Per-epoch weighted sampling across difficulty buckets.

    The sampler holds, for each bucket name, the dataset indices belonging to
    it. The CurriculumCallback writes the current (epoch's) bucket weight dict
    to `self.weights`. Each `__iter__()` returns `length` total indices, where
    bucket `b` contributes `round(weights[b] / sum(weights) * length)` of them
    sampled with replacement from that bucket. Indices are then shuffled.

    A bucket assigned weight 0 is excluded entirely (no representation that
    epoch). This is what enables true scaffolding: easy disappears once the
    model has used it.
    """

    def __init__(
        self,
        bucket_indices: dict[str, np.ndarray],
        length: int,
        initial_weights: dict[str, float] | None = None,
    ):
        self.bucket_indices = {b: np.asarray(idx) for b, idx in bucket_indices.items()}
        self.length = length
        self.weights = dict(initial_weights) if initial_weights else {b: 1.0 for b in bucket_indices}
        self._gen = np.random.default_rng()

    def __iter__(self):
        # Filter to buckets that actually exist + have non-zero weight.
        active = {
            b: w for b, w in self.weights.items()
            if w > 0 and b in self.bucket_indices and len(self.bucket_indices[b]) > 0
        }
        if not active:
            # Pathological: no active bucket. Fall back to uniform across all.
            active = {b: 1.0 for b in self.bucket_indices if len(self.bucket_indices[b]) > 0}
        total = sum(active.values())
        out: list[int] = []
        for b, w in active.items():
            n = int(round(w / total * self.length))
            if n == 0:
                continue
            sampled = self._gen.choice(self.bucket_indices[b], size=n, replace=True)
            out.extend(sampled.tolist())
        # Trim/pad to exactly self.length
        if len(out) > self.length:
            out = out[: self.length]
        elif len(out) < self.length:
            # Top up from the largest active bucket
            biggest = max(active, key=active.get)
            extra = self._gen.choice(self.bucket_indices[biggest], size=self.length - len(out), replace=True)
            out.extend(extra.tolist())
        self._gen.shuffle(out)
        return iter(out)

    def __len__(self):
        return self.length


class BucketCurriculumCallback(pl.Callback):
    """Schedule per-epoch bucket weights.

    `schedule` is a list of dicts; entry e (clamped to last entry if
    epoch >= len(schedule)) is the weight dict applied at the start of
    epoch e. Example for 6 explicit stages over 15 epochs:
        [{"easy": 1, "medium": 0, "hard": 0},
         {"easy": 0.5, "medium": 0.5, "hard": 0},
         {"easy": 0.2, "medium": 0.6, "hard": 0.2},
         {"easy": 0.1, "medium": 0.4, "hard": 0.5},
         {"easy": 0.0, "medium": 0.2, "hard": 0.8},
         {"easy": 0.0, "medium": 0.0, "hard": 1.0}]
    Epoch 5+ stays on the last stage (hard-only).
    """

    def __init__(self, sampler: "BucketCurriculumSampler", schedule: list[dict[str, float]]):
        super().__init__()
        self.sampler = sampler
        self.schedule = schedule

    def on_train_epoch_start(self, trainer, pl_module):
        e = trainer.current_epoch
        weights = self.schedule[min(e, len(self.schedule) - 1)]
        self.sampler.weights = dict(weights)
        for b, w in weights.items():
            pl_module.log(f"curriculum_w_{b}", float(w), on_epoch=True, on_step=False)
        wstr = " ".join(f"{b}={w:.2f}" for b, w in weights.items())
        print(f"[bucket-curriculum] epoch={e}  {wstr}")


class PARSeqSchedSample(PARSeq):
    """PARSeq + scheduled sampling on the decoder input (Bengio et al. 2015).

    During training, with probability `current_eps` per non-BOS, non-padding
    position, replaces the ground-truth token in `tgt_in` with the model's own
    greedy prediction at that position. This is the canonical "scheduled
    sampling" technique adapted for PARSeq's permuted decoder.

    The schedule (eps over epochs) is set externally by SchedSampleCallback.
    eps=0 reproduces vanilla PARSeq teacher-forcing exactly.

    Cost: one extra decoder pass per training step (to get the proxy
    predictions). At eps=0 we skip this and pay no overhead.
    """

    def __init__(self, eps_max: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.eps_max = float(eps_max)
        self.current_eps = 0.0  # updated per epoch by SchedSampleCallback
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)
        memory = self.model.encode(images)

        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        # Scheduled sampling: substitute model's own predictions into tgt_in
        # at eps fraction of positions (excluding BOS / pads / EOS).
        if self.current_eps > 0.0:
            with torch.no_grad():
                tgt_mask, query_mask = self.generate_attn_masks(tgt_perms[0])
                out = self.model.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask,
                    tgt_query_mask=query_mask,
                )
                pred_tokens = self.model.head(out).argmax(-1)  # [N, L]
            replace = torch.rand_like(tgt_in, dtype=torch.float) < self.current_eps
            replace[:, 0] = False  # never overwrite BOS
            replace = replace & ~tgt_padding_mask
            tgt_in = torch.where(replace, pred_tokens, tgt_in)
            # Recompute padding mask in case any predicted token equals pad/eos
            tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        # Standard PARSeq training body with the (possibly) substituted tgt_in.
        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.model.decode(
                tgt_in, memory, tgt_mask, tgt_padding_mask,
                tgt_query_mask=query_mask,
            )
            logits = self.model.head(out).flatten(end_dim=1)
            loss = loss + n * F.cross_entropy(
                logits, tgt_out.flatten(), ignore_index=self.pad_id,
            )
            loss_numel += n
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss = loss / loss_numel
        self.log("loss", loss)
        self.log("sched_sample_eps", self.current_eps)
        return loss


class GradualUnfreezeCallback(pl.Callback):
    """Progressively unfreeze model layers per epoch (ULMFiT-style).

    `schedule` is a list of {epoch, unfreeze} dicts, where `unfreeze` is a
    list of parameter-name prefixes. Special prefix "all" matches every
    parameter. At each epoch start the callback finds the latest stage with
    epoch <= current_epoch and sets requires_grad=True on parameters matching
    its prefixes; everything else gets requires_grad=False.

    Frozen params still occupy AdamW state but get None gradients so their
    weights don't update. No optimizer reinitialization needed.
    """

    def __init__(self, schedule: list[dict]):
        super().__init__()
        self.schedule = sorted(schedule, key=lambda s: int(s["epoch"]))
        self._last_active_epoch = None

    def _matches(self, name: str, prefixes: list[str]) -> bool:
        if "all" in prefixes:
            return True
        return any(name.startswith(p) for p in prefixes)

    def on_train_epoch_start(self, trainer, pl_module):
        e = trainer.current_epoch
        active = None
        for stage in self.schedule:
            if int(stage["epoch"]) <= e:
                active = stage
        if active is None:
            return
        if self._last_active_epoch == int(active["epoch"]):
            return  # no transition this epoch
        self._last_active_epoch = int(active["epoch"])

        prefixes = list(active["unfreeze"])
        n_train = 0
        n_freeze = 0
        for name, p in pl_module.named_parameters():
            if self._matches(name, prefixes):
                p.requires_grad = True
                n_train += p.numel()
            else:
                p.requires_grad = False
                n_freeze += p.numel()
        total = n_train + n_freeze
        pct = 100.0 * n_train / total if total else 0.0
        print(
            f"[gradual-unfreeze] epoch={e}  trainable={n_train/1e6:.2f}M "
            f"({pct:.1f}%)  frozen={n_freeze/1e6:.2f}M  prefixes={prefixes}"
        )


class StageEndCheckpointCallback(pl.Callback):
    """Save a tagged checkpoint at the end of each curriculum stage.

    `epochs[i]` -> `names[i]`: at the end of training for epoch `epochs[i]`,
    save the model state to `<ckpt_dir>/stage_end_<names[i]>_epoch<NN>.ckpt`.
    Independent of ModelCheckpoint's top-k pruning, so stage snapshots
    survive even if a later stage produces higher val_accuracy.
    """

    def __init__(self, ckpt_dir: Path, epochs: list[int], names: list[str]):
        super().__init__()
        if len(epochs) != len(names):
            raise ValueError("epochs and names must have equal length")
        self.ckpt_dir = Path(ckpt_dir)
        self._by_epoch = {int(e): n for e, n in zip(epochs, names)}

    def on_train_epoch_end(self, trainer, pl_module):
        e = trainer.current_epoch
        name = self._by_epoch.get(e)
        if name is None:
            return
        out = self.ckpt_dir / f"stage_end_{name}_epoch{e:02d}.ckpt"
        trainer.save_checkpoint(str(out))
        print(f"[stage-end-ckpt] saved {out}")


class SchedSampleCallback(pl.Callback):
    """Linear eps schedule for PARSeqSchedSample.

    eps stays at 0 for `warmup_epochs` (full teacher forcing), then ramps
    linearly from 0 to `eps_max` over the remaining epochs.
    """

    def __init__(self, eps_max: float, total_epochs: int, warmup_epochs: int = 1):
        super().__init__()
        self.eps_max = float(eps_max)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = max(0, int(warmup_epochs))

    def on_train_epoch_start(self, trainer, pl_module):
        e = trainer.current_epoch
        if e < self.warmup_epochs:
            eps = 0.0
        else:
            denom = max(1, self.total_epochs - self.warmup_epochs - 1)
            progress = (e - self.warmup_epochs) / denom
            eps = max(0.0, min(self.eps_max, self.eps_max * progress))
        pl_module.current_eps = eps
        pl_module.log("sched_sample_eps_epoch", eps, on_epoch=True, on_step=False)
        print(f"[sched-sample] epoch={e}  eps={eps:.4f}")


class PARSeqWithLengthHead(PARSeq):
    """PARSeq + auxiliary binary classifier predicting is_fixed_length.

    Training step replicates parent's body inline so the encoder pass is
    shared between the char loss and the length-mode head (single forward
    through the ViT encoder per training step).
    """

    def __init__(self, lengthmode_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        embed_dim = kwargs["embed_dim"]
        self.length_head = nn.Linear(embed_dim, 1)
        self.lengthmode_weight = float(lengthmode_weight)
        # Re-save hyperparameters to include the new ones.
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        images, labels, is_fixed = batch
        is_fixed = is_fixed.to(self.device).float()

        tgt = self.tokenizer.encode(labels, self._device)

        # Encode once, reuse for both heads.
        memory = self.model.encode(images)  # [N, P, D]

        # --- char loss (replicated from PARSeq.training_step) ---
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        char_loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.model.decode(
                tgt_in, memory, tgt_mask, tgt_padding_mask,
                tgt_query_mask=query_mask,
            )
            logits = self.model.head(out).flatten(end_dim=1)
            char_loss = char_loss + n * F.cross_entropy(
                logits, tgt_out.flatten(), ignore_index=self.pad_id,
            )
            loss_numel += n
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        char_loss = char_loss / loss_numel

        # --- length-mode loss (uses the same memory) ---
        pooled = memory.mean(dim=1)                  # [N, D]
        len_logit = self.length_head(pooled).squeeze(-1)
        len_loss = F.binary_cross_entropy_with_logits(len_logit, is_fixed)

        total = char_loss + self.lengthmode_weight * len_loss
        self.log("loss", char_loss)  # match parent's key name for char loss
        self.log("loss_lenmode", len_loss)
        self.log("loss_total", total)
        return total

    def validation_step(self, batch, batch_idx):
        images, labels, is_fixed = batch
        is_fixed = is_fixed.to(self.device).float()

        # Char metrics via parent's _eval_step (expects 2-tuple batch).
        # Internally calls forward → forward_logits_loss → encode again.
        # Validation cost isn't on the critical path, so the redundant encode
        # is fine here.
        result = self._eval_step((images, labels), True)
        self.outputs.append(result)

        # Length-mode accuracy.
        with torch.no_grad():
            memory = self.model.encode(images)
            pooled = memory.mean(dim=1)
            len_logit = self.length_head(pooled).squeeze(-1)
            len_pred = (len_logit > 0).float()
            len_acc = (len_pred == is_fixed).float().mean()
        self.log("val_lengthmode_acc", len_acc, on_step=False, on_epoch=True, prog_bar=True)
        return result


def _parseq_kwargs(cfg: dict) -> dict:
    return dict(
        charset_train=CHARSET,
        charset_test=CHARSET,
        max_label_length=cfg["max_label_length"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        warmup_pct=cfg["warmup_pct"],
        weight_decay=cfg["weight_decay"],
        img_size=tuple(cfg["img_size"]),
        patch_size=tuple(cfg["patch_size"]),
        embed_dim=cfg["embed_dim"],
        enc_num_heads=cfg["enc_num_heads"],
        enc_mlp_ratio=cfg["enc_mlp_ratio"],
        enc_depth=cfg["enc_depth"],
        dec_num_heads=cfg["dec_num_heads"],
        dec_mlp_ratio=cfg["dec_mlp_ratio"],
        dec_depth=cfg["dec_depth"],
        perm_num=cfg["perm_num"],
        perm_forward=cfg["perm_forward"],
        perm_mirrored=cfg["perm_mirrored"],
        decode_ar=cfg["decode_ar"],
        refine_iters=cfg["refine_iters"],
        dropout=cfg["dropout"],
    )


def build_model(cfg: dict):
    """Build PARSeq, with length head and/or scheduled sampling if configured.

    Mutually exclusive (only one supported at a time):
      cfg["length_head_enabled"] (default True): wraps PARSeq with
        PARSeqWithLengthHead and adds the auxiliary BCE loss.
      cfg["sched_sample"]["enabled"] (default False): wraps PARSeq with
        PARSeqSchedSample for scheduled-sampling supervision curriculum.
      Neither: returns vanilla PARSeq matching Model 1's setup.
    """
    sched_cfg = cfg.get("sched_sample", {}) or {}
    if bool(sched_cfg.get("enabled", False)):
        return PARSeqSchedSample(
            eps_max=float(sched_cfg.get("eps_max", 0.25)),
            **_parseq_kwargs(cfg),
        )
    if cfg.get("length_head_enabled", True):
        return PARSeqWithLengthHead(
            lengthmode_weight=cfg.get("lengthmode_weight", 0.1),
            **_parseq_kwargs(cfg),
        )
    return PARSeq(**_parseq_kwargs(cfg))


def main(cfg: dict) -> None:
    root = Path(cfg["data_dir"])
    img_size = tuple(cfg["img_size"])

    train_rows, val_rows = split_rows(root, cfg["val_frac"], cfg["seed"])
    print(f"train={len(train_rows)} val={len(val_rows)}")

    train_ds = CaptchaDir(root, img_size, rows=train_rows, augment=bool(cfg.get("augment", True)))
    val_ds = CaptchaDir(root, img_size, rows=val_rows, augment=False)

    curriculum_cfg = cfg.get("curriculum", {})
    curriculum_kind = curriculum_cfg.get("kind", "off")  # "off" | "d_max" | "buckets"
    # Back-compat: "enabled: true" without "kind" maps to the d_max curriculum.
    if curriculum_kind == "off" and bool(curriculum_cfg.get("enabled", False)):
        curriculum_kind = "d_max"
    length_head_enabled = bool(cfg.get("length_head_enabled", True))
    collate = captcha_collate if length_head_enabled else captcha_collate_no_lengthmode
    callbacks = []

    bs = cfg["batch_size"]
    epoch_len = (len(train_ds) // bs) * bs

    if curriculum_kind == "d_max":
        sampler = CurriculumSampler(
            train_ds.difficulties, length=epoch_len, d_max=1.0,
        )
        warmup = int(curriculum_cfg.get("warmup_epochs", 8))
        callbacks.append(CurriculumCallback(sampler, warmup))
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], sampler=sampler,
            num_workers=cfg["num_workers"], pin_memory=True,
            persistent_workers=cfg["num_workers"] > 0,
            collate_fn=collate, drop_last=True,
        )
        print(f"[curriculum] kind=d_max, warmup_epochs={warmup}")
    elif curriculum_kind == "buckets":
        bucket_indices = {
            b: np.where(train_ds.buckets == b)[0]
            for b in np.unique(train_ds.buckets)
        }
        schedule = curriculum_cfg["schedule"]
        if not schedule:
            raise ValueError("curriculum.schedule must be a non-empty list of weight dicts")
        sampler = BucketCurriculumSampler(
            bucket_indices, length=epoch_len, initial_weights=schedule[0],
        )
        callbacks.append(BucketCurriculumCallback(sampler, schedule))
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], sampler=sampler,
            num_workers=cfg["num_workers"], pin_memory=True,
            persistent_workers=cfg["num_workers"] > 0,
            collate_fn=collate, drop_last=True,
        )
        bucket_sizes = {b: len(idx) for b, idx in bucket_indices.items()}
        print(f"[curriculum] kind=buckets, sizes={bucket_sizes}, n_stages={len(schedule)}")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], pin_memory=True,
            persistent_workers=cfg["num_workers"] > 0,
            collate_fn=collate, drop_last=True,
        )
        print("[curriculum] disabled")

    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
        collate_fn=collate,
    )

    model = build_model(cfg)

    # Optional: load weights from a prior checkpoint (fine-tuning).
    finetune_from = cfg.get("finetune_from")
    if finetune_from:
        state = torch.load(finetune_from, map_location="cpu", weights_only=False)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        result = model.load_state_dict(state_dict, strict=False)
        print(f"[finetune] loaded {finetune_from}")
        if result.missing_keys:
            mk = result.missing_keys
            print(f"[finetune] missing keys ({len(mk)}): {mk[:10]}{'...' if len(mk)>10 else ''}")
        if result.unexpected_keys:
            uk = result.unexpected_keys
            print(f"[finetune] unexpected keys ({len(uk)}): {uk[:10]}{'...' if len(uk)>10 else ''}")

    # Optional: gradual unfreezing schedule (ULMFiT-style).
    gu_cfg = cfg.get("gradual_unfreeze") or {}
    if gu_cfg.get("schedule"):
        callbacks.append(GradualUnfreezeCallback(gu_cfg["schedule"]))
        print(f"[gradual-unfreeze] enabled, n_stages={len(gu_cfg['schedule'])}")

    # Hook scheduled-sampling callback to the matching model variant.
    sched_cfg = cfg.get("sched_sample", {}) or {}
    if bool(sched_cfg.get("enabled", False)):
        callbacks.append(SchedSampleCallback(
            eps_max=float(sched_cfg.get("eps_max", 0.25)),
            total_epochs=int(cfg["max_epochs"]),
            warmup_epochs=int(sched_cfg.get("warmup_epochs", 1)),
        ))
        print(f"[sched-sample] enabled, eps_max={sched_cfg.get('eps_max', 0.25)}, "
              f"warmup_epochs={sched_cfg.get('warmup_epochs', 1)}")

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="parseq-{epoch:02d}-{val_accuracy:.3f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(ckpt_cb)

    sec_cfg = cfg.get("stage_end_checkpoint") or {}
    if sec_cfg.get("enabled", False):
        callbacks.append(StageEndCheckpointCallback(
            ckpt_dir=ckpt_dir,
            epochs=list(sec_cfg["epochs"]),
            names=list(sec_cfg["names"]),
        ))
        print(f"[stage-end-ckpt] enabled, n={len(sec_cfg['epochs'])}")

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision=cfg["precision"],
        gradient_clip_val=cfg.get("gradient_clip_val", 20),
        callbacks=callbacks,
        log_every_n_steps=50,
        default_root_dir=cfg["ckpt_dir"],
    )

    trainer.fit(model, train_loader, val_loader)
    print("best val_accuracy:", ckpt_cb.best_model_score)
    print("best ckpt:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/parseq.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    main(cfg)
