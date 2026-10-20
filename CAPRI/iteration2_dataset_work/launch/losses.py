"""Iteration 2 losses: one loss per head (logged); the global loss weights the groups by learned uncertainty.
Null labels (-1) are masked, never 0. Ordinal heads: cumulative BCE on "level >= k" (k = 1..10), on rows with level > 0
only for effect / text heads (presence carries level 0), on every labelled row for font heads (0 is a real value).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = ["eff_presence", "eff_ordinal", "ctc", "length", "txt_presence", "txt_ordinal", "font"]
K = torch.arange(1, 11)


class FixedWeights(nn.Module):
    """Equal, constant weights for the seven averaged task groups."""
    def __init__(self):
        super().__init__()
        self.register_buffer("log_var", torch.zeros(len(GROUPS)))

    def forward(self, g):
        return sum(g[n] for n in GROUPS)

class UncertaintyWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(len(GROUPS)))

    def forward(self, g):
        return sum(torch.exp(-self.log_var[i]) * g[n] + self.log_var[i] for i, n in enumerate(GROUPS))


def presence_loss(logit, y):
    """-> per-head mean BCE, counts."""
    v = (y >= 0).float()
    l = F.binary_cross_entropy_with_logits(logit.float(), y.clamp(min=0).float(), reduction="none") * v
    return l.sum(0) / v.sum(0).clamp(min=1), v.sum(0)


def ordinal_loss(logits, level, valid, weight=None):
    """logits B x H x 10, level B x H, valid B x H (bool) -> per-head mean cumulative BCE, counts.
    weight: optional H x 11 table (per head and level) multiplying each row's loss (level balancing)."""
    tgt = (level.unsqueeze(-1) >= K.to(level.device)).float()
    w = valid.float()
    if weight is not None:
        w = w * weight[torch.arange(level.shape[1], device=level.device).view(1, -1), level.clamp(0, 10)]
    l = F.binary_cross_entropy_with_logits(logits.float(), tgt, reduction="none").mean(-1) * w
    n = valid.float().sum(0)
    return l.sum(0) / w.sum(0).clamp(min=1e-6), n


def ctc_loss(logits, chars, length, cols):
    lp = logits.float().log_softmax(-1).transpose(0, 1)  # T x B x 63
    tgt = chars.long() + 1
    flat = torch.cat([tgt[b, :length[b]] for b in range(tgt.shape[0])])
    return F.ctc_loss(lp, flat, cols.long(), length.long(), blank=0, reduction="mean", zero_infinity=True)


def level_weights(level, valid_min=1):
    """level: N x H numpy (labels) -> H x 11 tensor 1 / frequency of each level (rows with level >= valid_min),
    normalised to mean 1 over the present levels and clipped to [0.25, 4]."""
    import numpy as np
    H = level.shape[1]
    w = np.ones((H, 11), np.float32)
    for h in range(H):
        lv = level[:, h]
        c = np.bincount(lv[lv >= valid_min].clip(0, 10), minlength=11).astype(np.float64)
        m = c > 0
        if m.any():
            inv = np.where(m, 1.0 / np.maximum(c, 1), 0)
            inv = inv / inv[m].mean()
            w[h] = np.clip(np.where(m, inv, 1.0), 0.25, 4.0)
    return torch.from_numpy(w)


def compute(out, b, eff_w=None, txt_w=None):
    per = {}
    per["eff_presence"] = presence_loss(out["eff_presence"], b["eff_present"])
    per["eff_ordinal"] = ordinal_loss(out["eff_ordinal"], b["eff_level"].long(), b["eff_level"] > 0, eff_w)
    per["txt_presence"] = presence_loss(out["txt_presence"], b["txt_present"])
    per["txt_ordinal"] = ordinal_loss(out["txt_ordinal"], b["txt_level"].long(), b["txt_level"] > 0, txt_w)
    f = b["font"].long()
    fo = f[:, 2:6]
    per["font_ordinal"] = ordinal_loss(out["font_ordinal"], fo.clamp(min=0), fo >= 0)
    g = {}
    for k in ("eff_presence", "eff_ordinal", "txt_presence", "txt_ordinal"):
        l, n = per[k]
        g[k] = (l * (n > 0)).sum() / (n > 0).sum().clamp(min=1)
    f_ = out["ctc"].shape[1] // (b["image"].shape[-1] // 8)  # CTC frames per token column
    g["ctc"] = ctc_loss(out["ctc"], b["chars"], b["length"], b["cols"] * f_)
    g["length"] = F.cross_entropy(out["length"].float(), (b["length"].long() - 2).clamp(0, 6))
    cat = F.cross_entropy(out["font_cat"].float(), f[:, 0], ignore_index=-1) if (f[:, 0] >= 0).any() else out["font_cat"].sum() * 0
    stroke = F.cross_entropy(out["font_stroke"].float(), f[:, 1], ignore_index=-1) if (f[:, 1] >= 0).any() else out["font_stroke"].sum() * 0
    ital = F.binary_cross_entropy_with_logits(out["font_italic"].float(), f[:, 6].clamp(min=0).float())
    fl, fn = per["font_ordinal"]
    g["font"] = (cat + stroke + ital + (fl * (fn > 0)).sum() / (fn > 0).sum().clamp(min=1)) / 4
    per["font_parts"] = dict(cat=cat.detach(), stroke=stroke.detach(), italic=ital.detach())
    return g, per
