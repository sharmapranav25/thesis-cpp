"""Captcha dataset for Qwen2-VL fine-tuning.

Reads PHP-MyBB synth from `data/synth_mybb/labels.csv` and yields fully
processed model inputs with the user-turn portion of `labels` masked to -100
so loss is computed only on the 5-char assistant response.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


PROMPT = (
    "This is a 5-character captcha. The characters are case-sensitive and from "
    "the set [a-zA-Z0-9]. Output ONLY the 5 characters, nothing else, no "
    "explanation, no spaces."
)


def _user_messages(prompt: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]


def _full_messages(prompt: str, label: str) -> list[dict]:
    return _user_messages(prompt) + [{
        "role": "assistant",
        "content": [{"type": "text", "text": label}],
    }]


class CaptchaDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        processor,
        max_samples: Optional[int] = None,
        prompt: str = PROMPT,
    ):
        self.dir = Path(data_dir)
        self.processor = processor
        self.prompt = prompt
        rows = []
        with open(self.dir / "labels.csv", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append((row["filename"], row["label"]))
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        fn, label = self.rows[idx]
        img = Image.open(self.dir / fn).convert("RGB")

        full_text = self.processor.apply_chat_template(
            _full_messages(self.prompt, label),
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_text = self.processor.apply_chat_template(
            _user_messages(self.prompt),
            tokenize=False,
            add_generation_prompt=True,
        )

        full = self.processor(
            text=[full_text], images=[img], return_tensors="pt", padding=True
        )
        prefix = self.processor(
            text=[prefix_text], images=[img], return_tensors="pt", padding=True
        )
        prefix_len = prefix["input_ids"].shape[1]

        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[attention_mask == 0] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": full["pixel_values"],
            "image_grid_thw": full["image_grid_thw"],
        }
        return item


def collate_single(batch: list[dict]) -> dict:
    """Batch=1 collate that just unwraps the list."""
    assert len(batch) == 1
    item = batch[0]
    return {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "labels": item["labels"].unsqueeze(0),
        "pixel_values": item["pixel_values"],
        "image_grid_thw": item["image_grid_thw"],
    }
