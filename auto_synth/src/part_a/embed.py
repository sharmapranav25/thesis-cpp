"""DINOv2 embedding for character crops.

Crops are resized to 224x224, normalized, batched through DINOv2-small. Returns
[N, D] float32 array of CLS-token features. Loads model once, embeds in batches.
"""
from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel


class DinoEmbedder:
    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "cuda", batch_size: int = 64):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.tf = T.Compose([
            T.Lambda(lambda im: im.convert("RGB")),
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.batch_size = batch_size
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        out = np.empty((len(crops), self.dim), dtype=np.float32)
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i:i + self.batch_size]
            x = torch.stack([self.tf(c) for c in batch]).to(self.device)
            feats = self.model(x).last_hidden_state[:, 0]  # CLS token
            feats = torch.nn.functional.normalize(feats, dim=-1)
            out[i:i + len(batch)] = feats.cpu().numpy()
        return out
