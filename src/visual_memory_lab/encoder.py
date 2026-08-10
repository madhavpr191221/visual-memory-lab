"""Frozen CLIP encoder used by the visual-memory index."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested Torch device, preferring CUDA for ``auto``."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _json_value(value: Any) -> Any:
    """Convert processor configuration values into JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ClipEncoder:
    """Encode images and text with the pinned CLIP ViT-B/32 checkpoint."""

    model_id = MODEL_ID
    model_revision = MODEL_REVISION

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        self.model = CLIPModel.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.projection_dim)

    @property
    def processor_config(self) -> dict[str, object]:
        image_processor = self.processor.image_processor
        return {
            "size": _json_value(image_processor.size),
            "crop_size": _json_value(image_processor.crop_size),
            "resample": _json_value(image_processor.resample),
            "image_mean": _json_value(image_processor.image_mean),
            "image_std": _json_value(image_processor.image_std),
            "do_resize": bool(image_processor.do_resize),
            "do_center_crop": bool(image_processor.do_center_crop),
            "do_rescale": bool(image_processor.do_rescale),
            "do_normalize": bool(image_processor.do_normalize),
            "do_convert_rgb": bool(image_processor.do_convert_rgb),
        }

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(p=2, dim=-1, keepdim=True)

    @staticmethod
    def _feature_tensor(output: object) -> torch.Tensor:
        """Accept tensor returns from Transformers 5.9 and wrappers from 5.15."""

        if isinstance(output, torch.Tensor):
            return output
        pooled_output = getattr(output, "pooler_output", None)
        if isinstance(pooled_output, torch.Tensor):
            return pooled_output
        raise TypeError("CLIP feature extraction did not return a pooled tensor")

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        images: list[Image.Image] = []
        try:
            for path in image_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt").to(
                self.device
            )
            with torch.inference_mode():
                features = self._feature_tensor(
                    self.model.get_image_features(**inputs)
                )
                features = self._normalize(features)
            return features.detach().cpu().to(torch.float32).numpy()
        finally:
            for image in images:
                image.close()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(texts),
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            features = self._feature_tensor(self.model.get_text_features(**inputs))
            features = self._normalize(features)
        return features.detach().cpu().to(torch.float32).numpy()
