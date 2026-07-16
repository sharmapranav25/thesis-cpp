"""5-box regressor for fixed-length captchas.

Tiny ConvNet (~500K params) that maps 200×60 RGB → 5 axis-aligned bboxes
in [0,1] normalised coords (cx, cy, w, h). Sorted left-to-right by cx
in the loss to avoid permutation ambiguity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(in_c: int, out_c: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class CharBoxNet(nn.Module):
    """In: [B, 3, H, W]. Out: [B, 5, 4] of (cx, cy, w, h) ∈ (0, 1)."""

    def __init__(self, n_boxes: int = 5):
        super().__init__()
        self.n = n_boxes
        self.stem = nn.Sequential(
            _block(3, 32, 1),
            _block(32, 32, 1),
            _block(32, 64, 2),     # 100x30
            _block(64, 64, 1),
            _block(64, 128, 2),    # 50x15
            _block(128, 128, 1),
            _block(128, 256, 2),   # 25x8
            _block(256, 256, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(256, 4 * n_boxes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)
        f = self.pool(f)
        out = self.head(f)
        out = torch.sigmoid(out)  # bound to (0, 1)
        return out.view(-1, self.n, 4)


# ---------- bbox helpers ----------------------------------------------------

def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """[..., 4] cxcywh → [..., 4] x0y0x1y1."""
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def giou_loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """Per-box GIoU loss = 1 - GIoU. Inputs shape [..., 4] in (x0,y0,x1,y1)."""
    px0, py0, px1, py1 = pred_xyxy.unbind(-1)
    gx0, gy0, gx1, gy1 = gt_xyxy.unbind(-1)

    pw = (px1 - px0).clamp(min=0)
    ph = (py1 - py0).clamp(min=0)
    gw = (gx1 - gx0).clamp(min=0)
    gh = (gy1 - gy0).clamp(min=0)
    p_area = pw * ph
    g_area = gw * gh

    ix0 = torch.maximum(px0, gx0)
    iy0 = torch.maximum(py0, gy0)
    ix1 = torch.minimum(px1, gx1)
    iy1 = torch.minimum(py1, gy1)
    iw = (ix1 - ix0).clamp(min=0)
    ih = (iy1 - iy0).clamp(min=0)
    inter = iw * ih
    union = p_area + g_area - inter + 1e-9
    iou = inter / union

    cx0 = torch.minimum(px0, gx0)
    cy0 = torch.minimum(py0, gy0)
    cx1 = torch.maximum(px1, gx1)
    cy1 = torch.maximum(py1, gy1)
    c_area = (cx1 - cx0).clamp(min=0) * (cy1 - cy0).clamp(min=0) + 1e-9
    giou = iou - (c_area - union) / c_area
    return 1 - giou


def sort_by_cx(boxes: torch.Tensor) -> torch.Tensor:
    """[B, N, 4] cxcywh → sorted by cx ascending."""
    cx = boxes[..., 0]
    idx = cx.argsort(dim=-1)
    idx_exp = idx.unsqueeze(-1).expand_as(boxes)
    return torch.gather(boxes, -2, idx_exp)
