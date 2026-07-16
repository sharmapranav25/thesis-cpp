"""Character classifier on cropped single chars.

Multi-task model on shared ResNet-18 backbone:
  - head A (62-class): exact char identity (backup signal for ensemble)
  - head B (3-class):  case judgement {UPPER, LOWER, DIGIT}
                       — focuses capacity on the bias mode that dominates
                       PARSeq's residual error.

Input 64×64 RGB. Charset is the standard case-sensitive alphanumeric set.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHARSET)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}

# 3-way case label
CASE_UPPER = 0
CASE_LOWER = 1
CASE_DIGIT = 2
NUM_CASES = 3


def char_to_case(ch: str) -> int:
    if ch.isdigit():
        return CASE_DIGIT
    if ch.isupper():
        return CASE_UPPER
    return CASE_LOWER


# Precomputed: case index per char-id position (62 → 3)
CHAR_CASE = [char_to_case(c) for c in CHARSET]


class CharClsMultiTask(nn.Module):
    """ResNet-18 backbone, 2 heads."""

    def __init__(self, num_classes: int = NUM_CLASSES, num_cases: int = NUM_CASES):
        super().__init__()
        backbone = timm.create_model("resnet18", pretrained=False, num_classes=0, in_chans=3)
        self.backbone = backbone
        self.feat_dim = backbone.num_features
        self.head_letter = nn.Linear(self.feat_dim, num_classes)
        self.head_case = nn.Linear(self.feat_dim, num_cases)

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)
        return self.head_letter(f), self.head_case(f)


def build_charcls() -> CharClsMultiTask:
    return CharClsMultiTask()
