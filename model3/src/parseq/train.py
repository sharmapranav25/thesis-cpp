"""PARSeq training for Model 3 — adaptive curriculum (no teacher).

Forks Model 2's bucket-curriculum trainer and replaces teacher-defined static
buckets with student-defined dynamic buckets. Per-sample training loss is
tracked as an EMA across mini-epochs; samples are bucketed into
trivial / learning-band / too-hard by quantile of that EMA every cycle. The
next cycle samples 70% from the learning band, 15% from trivial, and 15%
from too-hard. Backtracking on validation regression and a curriculum-
saturation criterion replace Model 2's fixed-budget schedule.

Key components vs Model 2's train.py:

  M3CaptchaDir              returns (img, label, idx) so per-sample loss can
                            be tied back to the dataset row
  m3_collate                stacks images, lists labels, returns idx tensor
  AdaptiveCurriculumSampler holds per-sample EMA + dynamic quantile buckets;
                            yields cycle_size weighted indices per __iter__
  PARSeqWithLossLog         PARSeq subclass; training_step replicates parent
                            body and additionally captures per-sample CE on
                            the identity permutation, storing the array on
                            the module for the EMA callback to consume
  LossEmaUpdateCallback     on_train_batch_end -> sampler.update_ema
  RebucketCallback          on_train_epoch_end -> sampler.recompute_buckets
                            and log the new partition + stability metric
  BacktrackCallback         on_validation_end -> if val_word drops more than
                            threshold below best, reload best ckpt weights
                            and force the sampler to skip too-hard for the
                            next N cycles
  SaturationStopCallback    on_validation_end -> stop when val plateau over
                            plateau_window cycles AND bucket partition is
                            stable (mean movement below threshold)

Vanilla PARSeq is the only model variant supported here (no length head, no
scheduled sampling). Length-mode plumbing from M2 is dropped — M3 inherits
M2's lesson that length-aware variants regress.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Anaconda-base venv on Windows: the venv inherits Anaconda's stdlib but the
# native DLLs (sqlite3.dll, etc.) live under Library/bin and aren't on PATH by
# default. Mirrors M1/M2.
_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

# Reuse Model 2's vendored PARSeq (no need to copy third_party into M3).
_M3_ROOT = Path(__file__).resolve().parents[2]
_PARSEQ_CANDIDATES = [
    _M3_ROOT.parent / "model 2" / "third_party" / "parseq",
    _M3_ROOT.parent / "model 1" / "third_party" / "parseq",
    _M3_ROOT / "third_party" / "parseq",
]
PARSEQ_ROOT = next((p for p in _PARSEQ_CANDIDATES if p.exists()), None)
if PARSEQ_ROOT is None:
    raise RuntimeError(
        "Could not locate third_party/parseq under any of: "
        + ", ".join(str(p) for p in _PARSEQ_CANDIDATES)
    )
sys.path.insert(0, str(PARSEQ_ROOT))

import numpy as np
import torch
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
    """Heavy training-time augmentation. Same as M1/M2 — non-negotiable when
    the curriculum samples small slices of the pool with replacement.
    `augment: false` collapses training to ~1.30% val_word in M2."""
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class M3CaptchaDir(Dataset):
    """Read labels.csv into (image, label, idx) triples.

    Accepts the same 2 / 4 / 5-column CSV layouts as M2's CaptchaDir.
    The 5th column ("bucket": easy / medium / hard) is M1.5 v2's per-sample
    NLL tertile assignment — used by the SPCL hybrid for teacher-bucket
    cycles before the curriculum hands off to student-derived buckets at
    a configured cycle. Exposed via self.buckets (numpy array of strings).
    """

    def __init__(
        self,
        root: Path,
        img_size: tuple[int, int],
        rows: list[tuple] | None = None,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.img_size = img_size  # (H, W)
        if rows is None:
            rows = _read_rows(self.root)
        self.rows = rows
        self.buckets = np.array([r[4] for r in rows])  # M1.5 NLL tertile per row
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
        fn, lbl, *_ = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.aug is not None:
            arr = np.array(img)
            arr = self.aug(image=arr)["image"]
            img = Image.fromarray(arr)
        img = self.tf(img)
        return img, lbl, i


def m3_collate(batch):
    imgs, labels, indices = zip(*batch)
    return (
        torch.stack(imgs, 0),
        list(labels),
        torch.tensor(indices, dtype=torch.long),
    )


def _read_rows(root: Path) -> list[tuple]:
    """Read labels.csv; pad rows to 5-tuple (filename,label,difficulty,is_fixed,bucket)."""
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


# ---------------------------------------------------------------------------
# Adaptive sampler
# ---------------------------------------------------------------------------


# Bucket codes (int8 fits comfortably; -1 = unassigned).
# Codes are shared between teacher and student buckets — they index by
# difficulty rank: 0=easiest, 1=medium, 2=hardest, regardless of source.
_TRIVIAL = 0
_LEARNING = 1
_TOO_HARD = 2
_STUDENT_NAMES = {_TRIVIAL: "trivial", _LEARNING: "learning", _TOO_HARD: "too_hard"}
_TEACHER_NAMES = {_TRIVIAL: "easy", _LEARNING: "medium", _TOO_HARD: "hard"}
_BUCKET_NAMES = _STUDENT_NAMES  # back-compat alias used in logging
_TEACHER_STR_TO_CODE = {"easy": _TRIVIAL, "medium": _LEARNING, "hard": _TOO_HARD}


class AdaptiveCurriculumSampler(Sampler[int]):
    """Per-sample EMA loss → quantile buckets → phase-weighted sampling.

    State:
      loss_ema[i]            EMA of per-sample CE for row i (NaN until seen)
      seen_count[i]          how many times row i has been sampled+trained on
      bucket_assignments[i]  current bucket code (-1, 0, 1, 2)
      prev_assignments[i]    last cycle's bucket assignments (for stability)
      skip_too_hard          countdown of cycles to drop the too_hard slice
                             (set by BacktrackCallback after a val regression)

    Cold start: when no sample has an EMA yet, __iter__ falls back to uniform
    random sampling from the entire dataset. That gives the first cycle a
    seed of per-sample loss values; from cycle 1 onward, recompute_buckets()
    runs on the actual EMA values.

    Unseen samples after the first recompute are placed in the learning band
    rather than excluded — we want them to get drawn so they accumulate
    EMA. Once seen, their actual quantile assignment kicks in.

    Phased weights (M3 v0.3, 2026-05-04):
      `phases` is a list of dicts. Each phase has a weight per bucket and an
      optional `until_cycle` upper bound. The first phase whose
      `until_cycle` exceeds self.cycle is active; the last phase (no
      `until_cycle`) is the default. Mirrors M2's 1/1/13 schedule with
      student-defined buckets in place of teacher-defined ones.
    """

    def __init__(
        self,
        n_samples: int,
        cycle_size: int,
        ema_alpha: float = 0.6,
        phases: list[dict] | None = None,
        quantile_lo: float = 0.25,
        quantile_hi: float = 0.75,
        seed: int | None = None,
        teacher_buckets: np.ndarray | None = None,
    ):
        self.n = int(n_samples)
        self.cycle_size = int(cycle_size)
        self.ema_alpha = float(ema_alpha)
        self.phases = self._validate_phases(phases) if phases else [
            {"bucket_source": "student", "trivial": 0.15, "learning": 0.70, "too_hard": 0.15},
        ]
        self.q_lo = float(quantile_lo)
        self.q_hi = float(quantile_hi)
        if not (0.0 < self.q_lo < self.q_hi < 1.0):
            raise ValueError(f"quantile bounds invalid: {self.q_lo} {self.q_hi}")
        # Teacher buckets are static int codes per row (0=easy, 1=medium, 2=hard),
        # populated once from the M1.5 NLL labels.csv. Optional — absent means
        # only student-source phases are usable.
        self.teacher_buckets: np.ndarray | None = None
        if teacher_buckets is not None:
            tb = np.asarray(teacher_buckets)
            if tb.shape != (self.n,):
                raise ValueError(f"teacher_buckets shape {tb.shape} != ({self.n},)")
            self.teacher_buckets = tb.astype(np.int8)

        self.loss_ema = np.full(self.n, np.nan, dtype=np.float64)
        self.seen_count = np.zeros(self.n, dtype=np.int64)
        self.bucket_assignments = np.full(self.n, -1, dtype=np.int8)
        self.prev_assignments = self.bucket_assignments.copy()
        self.last_thresholds: tuple[float, float] | None = None
        self.skip_too_hard = 0
        self.cycle = 0

        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_phases(phases: list[dict]) -> list[dict]:
        """Each phase is a dict with:
            bucket_source: "teacher" or "student" (default "student")
            until_cycle:   optional int; phase active while sampler.cycle < this
            weights:       keyed by bucket name appropriate to source —
                           "easy"/"medium"/"hard" for teacher,
                           "trivial"/"learning"/"too_hard" for student.

        Internally normalized to:
            {bucket_source, until_cycle?, w0, w1, w2}  with code-indexed weights
        so the sampler doesn't care about the source-specific names downstream.
        """
        if not phases:
            raise ValueError("phases list must be non-empty")
        teacher_keys = {"easy", "medium", "hard"}
        student_keys = {"trivial", "learning", "too_hard"}
        out: list[dict] = []
        for i, p in enumerate(phases):
            source = str(p.get("bucket_source", "student"))
            if source not in ("teacher", "student"):
                raise ValueError(f"phase {i}: bucket_source must be 'teacher' or 'student'")
            keyset = teacher_keys if source == "teacher" else student_keys
            extra = set(p.keys()) - keyset - {"bucket_source", "until_cycle"}
            if extra:
                raise ValueError(f"phase {i} has unexpected keys for {source} source: {extra}")

            if source == "teacher":
                w0 = float(p.get("easy", 0.0))
                w1 = float(p.get("medium", 0.0))
                w2 = float(p.get("hard", 0.0))
            else:
                w0 = float(p.get("trivial", 0.0))
                w1 = float(p.get("learning", 0.0))
                w2 = float(p.get("too_hard", 0.0))
            if w0 <= 0 and w1 <= 0 and w2 <= 0:
                raise ValueError(f"phase {i} has all-zero weights")
            entry: dict = {"bucket_source": source, "w0": w0, "w1": w1, "w2": w2}
            if "until_cycle" in p:
                entry["until_cycle"] = int(p["until_cycle"])
            out.append(entry)
        # The final phase is the fallthrough default — it must have no until_cycle.
        if "until_cycle" in out[-1]:
            raise ValueError("the last phase must have no until_cycle (fallthrough default)")
        return out

    def _resolve_phase(self) -> dict:
        """Return the active phase dict (with code-indexed weights and source)."""
        for phase in self.phases:
            if "until_cycle" in phase and self.cycle < phase["until_cycle"]:
                return phase
        return self.phases[-1]

    def _current_weights(self) -> dict[str, float]:
        """Backward-compat: return weights as a name-keyed dict using STUDENT names.

        (Teacher and student buckets share codes 0/1/2 by design, so naming is
        cosmetic. The active phase's source is in `_resolve_phase()['bucket_source']`.)
        """
        phase = self._resolve_phase()
        return {
            "trivial": phase["w0"],
            "learning": phase["w1"],
            "too_hard": phase["w2"],
        }

    # -- EMA update --------------------------------------------------------

    def update_ema(self, indices: np.ndarray, losses: np.ndarray) -> None:
        if len(indices) != len(losses):
            raise ValueError("indices and losses length mismatch")
        a = self.ema_alpha
        for idx, l in zip(indices, losses):
            cur = self.loss_ema[idx]
            if np.isnan(cur):
                self.loss_ema[idx] = float(l)
            else:
                self.loss_ema[idx] = a * float(l) + (1.0 - a) * float(cur)
            self.seen_count[idx] += 1

    # -- Bucketing ---------------------------------------------------------

    def recompute_buckets(self) -> dict:
        """Snapshot current quantile-derived bucket membership.

        Returns a summary dict for logging."""
        seen = ~np.isnan(self.loss_ema)
        n_seen = int(seen.sum())

        self.prev_assignments = self.bucket_assignments.copy()

        if n_seen < 10:
            # Pathologically early — keep everyone unassigned-but-eligible.
            new = np.full(self.n, _LEARNING, dtype=np.int8)
            self.bucket_assignments = new
            self.last_thresholds = None
        else:
            seen_losses = self.loss_ema[seen]
            lo = float(np.quantile(seen_losses, self.q_lo))
            hi = float(np.quantile(seen_losses, self.q_hi))
            new = np.full(self.n, _LEARNING, dtype=np.int8)
            new[seen & (self.loss_ema < lo)] = _TRIVIAL
            new[seen & (self.loss_ema >= hi)] = _TOO_HARD
            # Unseen samples stay in _LEARNING so they get drawn.
            self.bucket_assignments = new
            self.last_thresholds = (lo, hi)

        sizes = {
            _BUCKET_NAMES[c]: int((self.bucket_assignments == c).sum())
            for c in (_TRIVIAL, _LEARNING, _TOO_HARD)
        }
        stab = self.partition_stability()
        self.cycle += 1
        return {
            "n_seen": n_seen,
            "sizes": sizes,
            "thresholds": self.last_thresholds,
            "partition_stability": stab,
            "ema_quantiles": self._ema_quantiles(),
        }

    def partition_stability(self) -> float:
        """Fraction of *previously-seen* samples that changed bucket since
        the last recompute. Returns 1.0 (max instability) when we don't
        have two valid snapshots yet."""
        seen = ~np.isnan(self.loss_ema)
        valid = seen & (self.prev_assignments >= 0) & (self.bucket_assignments >= 0)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return 1.0
        moved = int((self.bucket_assignments[valid] != self.prev_assignments[valid]).sum())
        return moved / n_valid

    def _ema_quantiles(self) -> dict[str, float] | None:
        seen = ~np.isnan(self.loss_ema)
        if seen.sum() == 0:
            return None
        v = self.loss_ema[seen]
        return {
            "min": float(v.min()),
            "p25": float(np.quantile(v, 0.25)),
            "p50": float(np.quantile(v, 0.50)),
            "p75": float(np.quantile(v, 0.75)),
            "max": float(v.max()),
        }

    # -- Sampling ----------------------------------------------------------

    def __iter__(self):
        phase = self._resolve_phase()
        source = phase["bucket_source"]

        # Pick the active bucket assignments array.
        if source == "teacher":
            if self.teacher_buckets is None:
                raise RuntimeError(
                    f"phase at cycle {self.cycle} requests teacher buckets, but "
                    f"sampler was constructed without teacher_buckets"
                )
            active = self.teacher_buckets
        else:
            # Student source: cold-start fallback to uniform if EMA is empty.
            if (self.bucket_assignments < 0).all():
                idx = self.rng.integers(0, self.n, size=self.cycle_size)
                return iter(idx.tolist())
            active = self.bucket_assignments

        weights = {0: float(phase["w0"]), 1: float(phase["w1"]), 2: float(phase["w2"])}
        if self.skip_too_hard > 0:
            weights[_TOO_HARD] = 0.0
            other_total = sum(v for k, v in weights.items() if k != _TOO_HARD and v > 0)
            if other_total > 0:
                for k in list(weights):
                    if k != _TOO_HARD:
                        weights[k] = weights[k] / other_total
            self.skip_too_hard -= 1

        out: list[int] = []
        for code in (_TRIVIAL, _LEARNING, _TOO_HARD):
            w = weights[code]
            if w <= 0:
                continue
            pool = np.where(active == code)[0]
            if len(pool) == 0:
                continue
            n_draw = int(round(w * self.cycle_size))
            if n_draw == 0:
                continue
            sampled = self.rng.choice(pool, size=n_draw, replace=True)
            out.extend(sampled.tolist())

        # Trim or top-up to exact cycle_size (rounding can leave us off by 1-2).
        if len(out) > self.cycle_size:
            out = out[: self.cycle_size]
        elif len(out) < self.cycle_size:
            biggest_code = max(
                (_TRIVIAL, _LEARNING, _TOO_HARD),
                key=lambda c: (active == c).sum(),
            )
            pool = np.where(active == biggest_code)[0]
            if len(pool) == 0:
                pool = np.arange(self.n)
            extra = self.rng.choice(pool, size=self.cycle_size - len(out), replace=True)
            out.extend(extra.tolist())

        self.rng.shuffle(out)
        return iter(out)

    def __len__(self):
        return self.cycle_size


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class LossEmaUpdateCallback(pl.Callback):
    """Reads pl_module._last_per_sample_loss + ._last_indices after each
    training batch and pushes them into the sampler's EMA. The model sets
    those fields inside its overridden training_step (see PARSeqWithLossLog)."""

    def __init__(self, sampler: AdaptiveCurriculumSampler):
        super().__init__()
        self.sampler = sampler

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        losses = getattr(pl_module, "_last_per_sample_loss", None)
        indices = getattr(pl_module, "_last_indices", None)
        if losses is None or indices is None:
            return
        self.sampler.update_ema(indices, losses)
        # Clear so we don't double-count on a stale read.
        pl_module._last_per_sample_loss = None
        pl_module._last_indices = None


class RebucketCallback(pl.Callback):
    """At the end of each train epoch (= end of one mini-epoch / cycle),
    snapshot the new bucket partition. Logs sizes, thresholds, partition
    stability, and EMA quantiles."""

    def __init__(self, sampler: AdaptiveCurriculumSampler):
        super().__init__()
        self.sampler = sampler
        self.history: list[dict] = []

    def on_train_epoch_end(self, trainer, pl_module):
        info = self.sampler.recompute_buckets()
        self.history.append(info)
        sizes = info["sizes"]
        stab = info["partition_stability"]
        thr = info["thresholds"]
        emaq = info["ema_quantiles"]
        thr_str = f"lo={thr[0]:.4f} hi={thr[1]:.4f}" if thr else "n/a"
        emaq_str = (
            f"p25={emaq['p25']:.4f} p50={emaq['p50']:.4f} p75={emaq['p75']:.4f}"
            if emaq else "no ema yet"
        )
        # Look ahead one cycle to the next active phase, so logged
        # next_weights reflect what cycle N+1 will actually use.
        prev_cycle = self.sampler.cycle
        self.sampler.cycle = prev_cycle  # cycle was just incremented inside recompute_buckets
        next_phase = self.sampler._resolve_phase()
        self.sampler.cycle = prev_cycle  # no-op restore (already)
        next_source = next_phase["bucket_source"]
        if next_source == "teacher":
            wstr = (
                f"easy={next_phase['w0']:.2f} "
                f"medium={next_phase['w1']:.2f} "
                f"hard={next_phase['w2']:.2f}"
            )
        else:
            wstr = (
                f"trivial={next_phase['w0']:.2f} "
                f"learning={next_phase['w1']:.2f} "
                f"too_hard={next_phase['w2']:.2f}"
            )
        print(
            f"[rebucket] cycle={trainer.current_epoch} "
            f"sizes={sizes} thresholds=({thr_str}) "
            f"stability={stab:.4f}  ema=[{emaq_str}]  "
            f"seen={info['n_seen']}/{self.sampler.n}  "
            f"next_source={next_source} next_weights=[{wstr}]"
        )
        # Log scalar metrics for trackers.
        pl_module.log("cur_trivial", float(sizes["trivial"]), on_epoch=True, on_step=False)
        pl_module.log("cur_learning", float(sizes["learning"]), on_epoch=True, on_step=False)
        pl_module.log("cur_too_hard", float(sizes["too_hard"]), on_epoch=True, on_step=False)
        pl_module.log("cur_partition_stability", float(stab), on_epoch=True, on_step=False)


class BacktrackCallback(pl.Callback):
    """If validation accuracy drops more than `val_drop_threshold` below the
    best-so-far, restore weights from the best ckpt and force the sampler
    to skip too-hard for `rollback_skip_too_hard` cycles.

    Bails out (sets trainer.should_stop=True) after
    `max_consecutive_backtracks` rollbacks without improvement.
    """

    def __init__(
        self,
        sampler: AdaptiveCurriculumSampler,
        ckpt_callback: ModelCheckpoint,
        val_drop_threshold: float = 0.015,
        rollback_skip_too_hard: int = 1,
        max_consecutive_backtracks: int = 3,
        monitor: str = "val_accuracy",
    ):
        super().__init__()
        self.sampler = sampler
        self.ckpt_callback = ckpt_callback
        self.val_drop_threshold = float(val_drop_threshold)
        self.rollback_skip_too_hard = int(rollback_skip_too_hard)
        self.max_consecutive_backtracks = int(max_consecutive_backtracks)
        self.monitor = monitor

        self.best_val: float = -float("inf")
        self.consecutive_backtracks: int = 0

    def _read_metric(self, trainer) -> float | None:
        v = trainer.callback_metrics.get(self.monitor)
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return float(v.item())
        return float(v)

    def on_validation_end(self, trainer, pl_module):
        cur = self._read_metric(trainer)
        if cur is None:
            return
        if cur > self.best_val:
            self.best_val = cur
            self.consecutive_backtracks = 0
            return

        drop = self.best_val - cur
        if drop <= self.val_drop_threshold:
            return  # within tolerance — leave model as is

        best_path = getattr(self.ckpt_callback, "best_model_path", "") or ""
        if not best_path or not Path(best_path).exists():
            print(
                f"[backtrack] val drop {drop:.4f} > {self.val_drop_threshold:.4f} "
                f"but no best ckpt available yet — skipping"
            )
            return

        print(
            f"[backtrack] val drop {drop:.4f} > {self.val_drop_threshold:.4f}; "
            f"restoring weights from {best_path}; "
            f"skip too-hard for {self.rollback_skip_too_hard} cycle(s); "
            f"clearing optimizer momentum"
        )
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        pl_module.load_state_dict(sd)
        # Without this, Adam's exp_avg / exp_avg_sq buffers from the cycle
        # that just regressed bias the next gradient step away from the
        # restored weights — backtrack recoveries failed in run 6 partly
        # for this reason. Clearing the per-param state forces Adam to
        # rebuild momentum from fresh gradients.
        for opt in trainer.optimizers:
            opt.state.clear()
        self.sampler.skip_too_hard = max(self.sampler.skip_too_hard, self.rollback_skip_too_hard)

        self.consecutive_backtracks += 1
        # Lightning forbids self.log() inside on_validation_end; print only.
        print(f"[backtrack] consecutive_backtracks={self.consecutive_backtracks}/{self.max_consecutive_backtracks}")
        if self.consecutive_backtracks >= self.max_consecutive_backtracks:
            print(
                f"[backtrack] {self.consecutive_backtracks} consecutive backtracks "
                f"with no improvement — stopping training"
            )
            trainer.should_stop = True


class SaturationStopCallback(pl.Callback):
    """Stop when both:
      (a) val accuracy has plateaued for `plateau_window` cycles
          (max - min over window < min_delta), AND
      (b) bucket partition is stable
          (mean partition_stability over window < stability_threshold).

    Single-criterion stalls are allowed — the goal is to detect when the
    *adaptive mechanism itself* has nothing left to do, not just when val
    accuracy temporarily flatlines.
    """

    def __init__(
        self,
        sampler: AdaptiveCurriculumSampler,
        plateau_window: int = 6,
        min_delta: float = 0.005,
        partition_stability_threshold: float = 0.05,
        monitor: str = "val_accuracy",
    ):
        super().__init__()
        self.sampler = sampler
        self.plateau_window = int(plateau_window)
        self.min_delta = float(min_delta)
        self.partition_stability_threshold = float(partition_stability_threshold)
        self.monitor = monitor

        self.val_history: list[float] = []
        self.stab_history: list[float] = []

    def _read_metric(self, trainer) -> float | None:
        v = trainer.callback_metrics.get(self.monitor)
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return float(v.item())
        return float(v)

    def on_validation_end(self, trainer, pl_module):
        cur = self._read_metric(trainer)
        if cur is None:
            return
        self.val_history.append(cur)
        # The sampler's stability reflects the most recent recompute_buckets;
        # RebucketCallback fires at on_train_epoch_end, which is BEFORE
        # validation_end, so the value here is up-to-date.
        self.stab_history.append(self.sampler.partition_stability())

        print(
            f"[cycle] N={trainer.current_epoch}  val_acc={cur:.4f}  "
            f"partition_stab={self.stab_history[-1]:.4f}"
        )

        if len(self.val_history) < self.plateau_window:
            return

        recent_val = self.val_history[-self.plateau_window:]
        val_delta = max(recent_val) - min(recent_val)
        val_plateau = val_delta < self.min_delta

        recent_stab = self.stab_history[-self.plateau_window:]
        avg_stab = sum(recent_stab) / len(recent_stab)
        partition_stable = avg_stab < self.partition_stability_threshold

        msg = (
            f"[saturation] cycle={trainer.current_epoch} "
            f"val_delta={val_delta:.4f} (min={self.min_delta:.4f}) "
            f"avg_stab={avg_stab:.4f} (min={self.partition_stability_threshold:.4f}) "
            f"plateau={val_plateau} stable={partition_stable}"
        )
        print(msg)

        if val_plateau and partition_stable:
            print("[saturation] both criteria met — stopping training")
            trainer.should_stop = True


# ---------------------------------------------------------------------------
# PARSeq subclass: capture per-sample loss in training_step
# ---------------------------------------------------------------------------


class PARSeqWithLossLog(PARSeq):
    """PARSeq with per-sample CE captured during training_step.

    Replicates the parent training_step body inline so we can grab the
    identity-permutation logits before they're flattened+reduced. Per-sample
    loss is the mean CE across non-pad positions; the array (length N) and
    the corresponding global dataset indices are stashed on the module as
    `_last_per_sample_loss` and `_last_indices` for the LossEmaUpdateCallback
    to consume on_train_batch_end.

    validation_step / test_step strip the index from the 3-tuple and delegate
    to the parent class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_per_sample_loss: np.ndarray | None = None
        self._last_indices: np.ndarray | None = None

    def training_step(self, batch, batch_idx):
        images, labels, indices = batch
        tgt = self.tokenizer.encode(labels, self._device)
        memory = self.model.encode(images)

        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

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

        # Per-sample AR loss for curriculum bucketing.
        # Teacher-forced training-step loss is a misleading bucketing signal:
        # the decoder receives ground-truth tokens at every position, so loss
        # can be very low even when autoregressive decoding compounds errors
        # and val_accuracy stays at 1%. Run an explicit AR forward (eval mode,
        # no_grad) and use *that* loss for the EMA. Cost: one extra forward
        # pass per train step (~30-50% overhead) — worth it for a bucketing
        # signal that matches what we ultimately measure.
        with torch.no_grad():
            was_training = self.training
            self.eval()
            try:
                ar_logits = self(images)  # [N, num_steps, V]
            finally:
                if was_training:
                    self.train()
            gt = tgt[:, 1:]
            L = gt.size(1)
            ar_logits_trunc = ar_logits[:, :L, :]
            N_ = ar_logits_trunc.size(0)
            V = ar_logits_trunc.size(-1)
            per_pos_ce = F.cross_entropy(
                ar_logits_trunc.reshape(-1, V),
                gt.reshape(-1),
                ignore_index=self.pad_id, reduction="none",
            ).view(N_, L)
            mask = (gt != self.pad_id).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            per_sample = (per_pos_ce * mask).sum(dim=1) / denom
            self._last_per_sample_loss = per_sample.detach().cpu().numpy()
            self._last_indices = indices.detach().cpu().numpy()

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, _ = batch
        return super().validation_step((images, labels), batch_idx)

    def test_step(self, batch, batch_idx):
        images, labels, _ = batch
        return super().test_step((images, labels), batch_idx)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


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


def build_model(cfg: dict) -> PARSeqWithLossLog:
    return PARSeqWithLossLog(**_parseq_kwargs(cfg))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: dict) -> None:
    root = Path(cfg["data_dir"])
    img_size = tuple(cfg["img_size"])

    train_rows, val_rows = split_rows(root, cfg["val_frac"], cfg["seed"])
    print(f"train={len(train_rows)} val={len(val_rows)}")

    train_ds = M3CaptchaDir(
        root, img_size, rows=train_rows,
        augment=bool(cfg.get("augment", True)),
    )
    val_ds = M3CaptchaDir(root, img_size, rows=val_rows, augment=False)

    cur_cfg = cfg.get("curriculum", {}) or {}
    if cur_cfg.get("mode", "adaptive") != "adaptive":
        raise ValueError(
            f"Model 3 only supports curriculum.mode=adaptive; got {cur_cfg.get('mode')}"
        )

    cycle_size = int(cur_cfg.get("cycle_size", 18000))
    bs = int(cfg["batch_size"])
    # Round cycle_size down to a multiple of batch_size for clean drop_last batching.
    cycle_size = (cycle_size // bs) * bs
    if cycle_size <= 0:
        raise ValueError(f"cycle_size must be >= batch_size; got {cur_cfg.get('cycle_size')} vs bs={bs}")

    bucket_cfg = cur_cfg.get("buckets", {}) or {}
    phases_cfg = cur_cfg.get("phases")
    weight_cfg = cur_cfg.get("sample_weights")
    if phases_cfg:
        phases = phases_cfg
    elif weight_cfg:
        # Back-compat: a single static student-source weights block.
        compat = dict(weight_cfg)
        compat.setdefault("bucket_source", "student")
        phases = [compat]
    else:
        phases = None  # sampler will use its built-in default

    # Build teacher buckets (if any phase uses teacher source).
    needs_teacher = any(
        str((p or {}).get("bucket_source", "student")) == "teacher"
        for p in (phases or [])
    )
    teacher_buckets_arr: np.ndarray | None = None
    if needs_teacher:
        unknown = [b for b in train_ds.buckets if b not in _TEACHER_STR_TO_CODE]
        if unknown:
            uniq = sorted(set(unknown))[:5]
            raise ValueError(
                f"teacher source requested but labels.csv has unknown bucket strings: {uniq}. "
                f"Expected one of {sorted(_TEACHER_STR_TO_CODE)}"
            )
        teacher_buckets_arr = np.array(
            [_TEACHER_STR_TO_CODE[b] for b in train_ds.buckets], dtype=np.int8,
        )
        sizes = {n: int((teacher_buckets_arr == c).sum()) for c, n in _TEACHER_NAMES.items()}
        print(f"[adaptive] teacher buckets loaded from labels.csv: {sizes}")

    sampler = AdaptiveCurriculumSampler(
        n_samples=len(train_ds),
        cycle_size=cycle_size,
        ema_alpha=float(cur_cfg.get("ema_alpha", 0.6)),
        phases=phases,
        quantile_lo=float(bucket_cfg.get("trivial_quantile", 0.25)),
        quantile_hi=float(bucket_cfg.get("too_hard_quantile", 0.75)),
        seed=int(cfg.get("seed", 0)),
        teacher_buckets=teacher_buckets_arr,
    )
    print(
        f"[adaptive] cycle_size={cycle_size}  ema_alpha={sampler.ema_alpha}  "
        f"quantiles=({sampler.q_lo}, {sampler.q_hi})  n_phases={len(sampler.phases)}"
    )
    for i, p in enumerate(sampler.phases):
        until = p.get("until_cycle", "inf")
        src = p["bucket_source"]
        if src == "teacher":
            wstr = f"easy={p['w0']:.2f} medium={p['w1']:.2f} hard={p['w2']:.2f}"
        else:
            wstr = f"trivial={p['w0']:.2f} learning={p['w1']:.2f} too_hard={p['w2']:.2f}"
        print(f"[adaptive] phase {i}: until_cycle={until}  source={src}  {wstr}")

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
        collate_fn=m3_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
        collate_fn=m3_collate,
    )

    model = build_model(cfg)

    # Optional fine-tune-from (parity with M2; not used in the flagship recipe).
    finetune_from = cfg.get("finetune_from")
    if finetune_from:
        state = torch.load(finetune_from, map_location="cpu", weights_only=False)
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        result = model.load_state_dict(sd, strict=False)
        print(f"[finetune] loaded {finetune_from}")
        if result.missing_keys:
            mk = result.missing_keys
            print(f"[finetune] missing keys ({len(mk)}): {mk[:10]}{'...' if len(mk) > 10 else ''}")
        if result.unexpected_keys:
            uk = result.unexpected_keys
            print(f"[finetune] unexpected keys ({len(uk)}): {uk[:10]}{'...' if len(uk) > 10 else ''}")

    # Build callbacks. ModelCheckpoint must run before BacktrackCallback so
    # best_model_path is up-to-date when we read it.
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

    bt_cfg = cfg.get("backtrack", {}) or {}
    es_cfg = cfg.get("early_stop", {}) or {}

    callbacks: list[pl.Callback] = [
        LossEmaUpdateCallback(sampler),
        RebucketCallback(sampler),
        ckpt_cb,
        BacktrackCallback(
            sampler=sampler,
            ckpt_callback=ckpt_cb,
            val_drop_threshold=float(bt_cfg.get("val_drop_threshold", 0.015)),
            rollback_skip_too_hard=int(bt_cfg.get("rollback_skip_too_hard", 1)),
            max_consecutive_backtracks=int(bt_cfg.get("max_consecutive_backtracks", 3)),
        ),
        SaturationStopCallback(
            sampler=sampler,
            plateau_window=int(es_cfg.get("plateau_window", 6)),
            min_delta=float(es_cfg.get("min_delta", 0.005)),
            partition_stability_threshold=float(
                es_cfg.get("partition_stability_threshold", 0.05),
            ),
        ),
    ]

    max_cycles = int(es_cfg.get("max_cycles", cfg.get("max_epochs", 25)))

    trainer = pl.Trainer(
        max_epochs=max_cycles,
        accelerator="gpu",
        devices=1,
        precision=cfg["precision"],
        gradient_clip_val=cfg.get("gradient_clip_val", 5),
        callbacks=callbacks,
        log_every_n_steps=20,
        default_root_dir=cfg["ckpt_dir"],
    )

    trainer.fit(model, train_loader, val_loader)

    print("best val_accuracy:", ckpt_cb.best_model_score)
    print("best ckpt:", ckpt_cb.best_model_path)
    print(f"final cycle: {trainer.current_epoch}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/parseq_adaptive.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    main(cfg)
