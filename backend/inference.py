"""
InferenceEngine: loads the SegFormer-B1 checkpoint once at FastAPI startup
and exposes a classify() method that turns raw image bytes into a full result.

Logic mirrors scripts/infer.py's core pipeline, minus CLI/file I/O plumbing.
Preprocessing parity with scripts/dataset.py's val pipeline was verified in
Phase 5 (max abs diff 4.8e-7).

Key gotchas from Phase 3 / 5:
  - torch.load(..., weights_only=False)  (Path objects in args)
  - ckpt["model"]  NOT  ckpt["model_state_dict"]
"""
from __future__ import annotations

import base64
import io
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.emissions import compute_emissions, EmissionsResult  # noqa: E402

log = logging.getLogger(__name__)


# ---------- constants ----------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = [
    "no_data",        # 0
    "background",     # 1
    "building",       # 2
    "road",           # 3
    "water",          # 4
    "barren",         # 5
    "forest",         # 6
    "agriculture",    # 7
]

PALETTE = {
    0: (0, 0, 0),
    1: (210, 210, 210),
    2: (220, 20, 60),
    3: (70, 70, 70),
    4: (30, 144, 255),
    5: (210, 180, 140),
    6: (34, 139, 34),
    7: (255, 215, 0),
}

TRAIN_RESOLUTION = 1024
MODEL_ID = "nvidia/segformer-b1-finetuned-ade-512-512"


@dataclass
class ClassifyResult:
    percentages: dict[str, float]
    emissions: EmissionsResult
    mask_png_base64: str
    inference_ms: int
    input_shape: tuple[int, int]
    warnings: list[str]


class InferenceEngine:
    """Singleton: load once, classify many."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        log.info("Loading SegFormer-B1 on %s from %s", self.device, self.checkpoint_path)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_ID,
            num_labels=8,
            ignore_mismatched_sizes=True,
        )

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "model" not in ckpt:
            raise KeyError(
                f"Expected 'model' key in checkpoint, got {list(ckpt.keys())}. "
                "Phase 3 save_checkpoint stores state_dict under ckpt['model']."
            )
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            log.warning("Missing keys on load: %d (first 5: %s)", len(missing), missing[:5])
        if unexpected:
            log.warning("Unexpected keys on load: %d (first 5: %s)", len(unexpected), unexpected[:5])

        self.model.to(self.device)
        self.model.eval()

        if "val_mIoU" in ckpt:
            log.info("Checkpoint val_mIoU: %.4f (epoch %s)",
                     ckpt["val_mIoU"], ckpt.get("epoch", "?"))

    @torch.no_grad()
    def classify(
        self,
        image_bytes: bytes,
        tta: bool = True,
        pixel_size_m: float = 0.3,
    ) -> ClassifyResult:
        t0 = time.perf_counter()
        warnings: list[str] = []

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_h, orig_w = img.height, img.width

        if (orig_h, orig_w) != (TRAIN_RESOLUTION, TRAIN_RESOLUTION):
            warnings.append(
                f"Input {orig_w}x{orig_h} resized to {TRAIN_RESOLUTION}x{TRAIN_RESOLUTION}; "
                f"model trained at 0.3m/pixel, GSD mismatch may degrade accuracy."
            )
            img = img.resize((TRAIN_RESOLUTION, TRAIN_RESOLUTION), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        logits = self._forward_upsampled(tensor)
        if tta:
            logits_h = self._forward_upsampled(torch.flip(tensor, dims=[3]))
            logits_h = torch.flip(logits_h, dims=[3])
            logits_v = self._forward_upsampled(torch.flip(tensor, dims=[2]))
            logits_v = torch.flip(logits_v, dims=[2])
            logits = (logits + logits_h + logits_v) / 3.0

        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        counts = np.bincount(mask.ravel(), minlength=8)
        total_pixels = int(counts.sum())
        pixel_area_m2 = pixel_size_m ** 2

        percentages = {
            CLASS_NAMES[i]: round(float(counts[i]) / total_pixels * 100.0, 4)
            for i in range(8)
        }

        class_pixel_counts = {i: int(counts[i]) for i in range(8)}
        emissions = compute_emissions(
            class_pixel_counts=class_pixel_counts,
            total_pixels=total_pixels,
            pixel_area_m2=pixel_area_m2,
        )

        mask_png_base64 = self._mask_to_base64_png(mask)

        inference_ms = int((time.perf_counter() - t0) * 1000)

        return ClassifyResult(
            percentages=percentages,
            emissions=emissions,
            mask_png_base64=mask_png_base64,
            inference_ms=inference_ms,
            input_shape=(orig_h, orig_w),
            warnings=warnings,
        )

    def _forward_upsampled(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits

    def _mask_to_base64_png(self, mask: np.ndarray) -> str:
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in PALETTE.items():
            rgb[mask == idx] = color
        buf = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
