"""GroundingDINO text-prompted bbox detection via HuggingFace transformers.

Wraps IDEA-Research/grounding-dino-tiny. Given an image and a free-text
prompt (e.g. "thin black serif letters with red noise lines."), returns
axis-aligned bounding boxes for every region matching the prompt.

The text prompt is normalized to GroundingDINO's expected format: lowercase,
trailing period.
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def _format_prompt(prompt: str) -> str:
    p = prompt.lower().strip()
    if not p.endswith("."):
        p += "."
    return p


class GroundingDinoDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        prompt: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> dict:
        """Returns {'boxes': [[x1,y1,x2,y2], ...], 'scores': [...], 'labels': [str, ...]}.

        Boxes are in pixel coords matching the input image's size.
        """
        prompt_norm = _format_prompt(prompt)
        inputs = self.processor(images=image, text=prompt_norm, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        raw_labels = results.get("text_labels", results.get("labels", []))
        if hasattr(raw_labels, "tolist"):
            raw_labels = raw_labels.tolist()
        labels = [str(l) for l in raw_labels]
        return {"boxes": boxes, "scores": scores, "labels": labels}
