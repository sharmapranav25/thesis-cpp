"""Run Qwen2-VL (base or LoRA-adapted) on a directory of captcha images.

Outputs CSV: filename,label,prediction,match
where label is parsed from the filename stem (eval-set convention).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from src.qwenvl.data import PROMPT, _user_messages


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="dir of images")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--lora", default=None, help="path to peft LoRA adapter dir; omit for zero-shot")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()

    if args.lora:
        from peft import PeftModel
        print(f"attaching LoRA: {args.lora}", flush=True)
        model = PeftModel.from_pretrained(model, args.lora)
        model.eval()

    text_prompt = processor.apply_chat_template(
        _user_messages(PROMPT), tokenize=False, add_generation_prompt=True
    )

    files = sorted([
        p for p in src.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])
    if args.limit:
        files = files[: args.limit]

    print(f"running on {len(files)} images", flush=True)
    rows = []
    t0 = time.time()

    @torch.inference_mode()
    def predict(img: Image.Image) -> str:
        inputs = processor(
            text=[text_prompt], images=[img], return_tensors="pt", padding=True
        ).to("cuda")
        ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        gen = ids[:, inputs["input_ids"].shape[1]:]
        out = processor.batch_decode(gen, skip_special_tokens=True)[0]
        out = out.strip().strip('"').strip("'")
        if " " in out or ":" in out:
            out = out.replace(":", " ").split()[-1]
        return out[:5]

    for i, fp in enumerate(files):
        img = Image.open(fp).convert("RGB")
        label = fp.stem  # filename = ground truth on the eval set
        try:
            pred = predict(img)
        except Exception as e:  # noqa: BLE001
            print(f"  error on {fp.name}: {e}", flush=True)
            pred = ""
        match = int(pred == label)
        rows.append({"filename": fp.name, "label": label, "prediction": pred, "match": match})
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            running = sum(r["match"] for r in rows) / len(rows)
            print(f"  {i+1}/{len(files)}  acc {running:.3f}  {dt/(i+1):.2f}s/img", flush=True)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label", "prediction", "match"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n = len(rows)
    word = sum(r["match"] for r in rows) / n
    char_correct = sum(
        sum(1 for a, b in zip(r["label"], r["prediction"]) if a == b)
        for r in rows
    )
    char_total = sum(len(r["label"]) for r in rows)
    char = char_correct / char_total
    print(f"\nfinal  word acc {word:.4f} ({word*100:.2f}%)  "
          f"char acc {char:.4f} ({char*100:.2f}%)  n={n}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
