"""Small, auditable IO and geometry helpers shared by every experiment stage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json_dump(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_jsonl_dump(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)


def atomic_torch_save(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def atomic_numpy_save(value: np.ndarray, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, destination)


def read_exr_r(path: str | Path) -> np.ndarray:
    import OpenEXR

    source = OpenEXR.File(str(path), separate_channels=True, header_only=False)
    try:
        channels = source.channels()
        if "R" not in channels:
            raise ValueError(f"R channel missing from OpenEXR depth: {path}")
        return np.ascontiguousarray(channels["R"].pixels, dtype=np.float32)
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()


def target_geometry(
    original_hw: tuple[int, int], canvas_hw: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    h0, w0 = original_hw
    height, width = canvas_hw
    if min(h0, w0, height, width) <= 0:
        raise ValueError("image and canvas dimensions must be positive")
    scale = min(height / h0, width / w0)
    h1 = min(height, max(1, round(scale * h0)))
    w1 = min(width, max(1, round(scale * w0)))
    top = (height - h1) // 2
    bottom = height - h1 - top
    left = (width - w1) // 2
    right = width - w1 - left
    return (h1, w1), (top, bottom, left, right)


def letterbox_rgb(
    rgb_u8: np.ndarray, canvas_hw: tuple[int, int] = (384, 512)
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("rgb_u8 must be uint8 [H,W,3]")
    (h1, w1), padding = target_geometry(tuple(rgb_u8.shape[:2]), canvas_hw)
    top, _, left, _ = padding
    source = torch.from_numpy(rgb_u8.copy()).permute(2, 0, 1).float().div_(255.0)
    resized = F.interpolate(
        source.unsqueeze(0), size=(h1, w1), mode="bilinear", align_corners=False,
        antialias=True,
    )[0].clamp_(0.0, 1.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    resized = (resized - mean) / std
    canvas = torch.zeros((3, *canvas_hw), dtype=torch.float32)
    content = torch.zeros(canvas_hw, dtype=torch.bool)
    canvas[:, top : top + h1, left : left + w1] = resized
    content[top : top + h1, left : left + w1] = True
    return canvas, content, padding


def letterbox_rgb_raw(
    rgb_u8: np.ndarray, canvas_hw: tuple[int, int] = (384, 512)
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """Letterbox RGB to ``[0,1]`` without ImageNet normalization.

    TOM-Net's official Torch7 input contract is raw RGB in ``[0,1]``.  This
    helper intentionally stays separate from :func:`letterbox_rgb`, which is
    the ImageNet-normalized DINOv3 path.
    """

    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("rgb_u8 must be uint8 [H,W,3]")
    (h1, w1), padding = target_geometry(tuple(rgb_u8.shape[:2]), canvas_hw)
    top, _, left, _ = padding
    source = torch.from_numpy(rgb_u8.copy()).permute(2, 0, 1).float().div_(255.0)
    resized = F.interpolate(
        source.unsqueeze(0), size=(h1, w1), mode="bilinear", align_corners=False,
        antialias=True,
    )[0].clamp_(0.0, 1.0)
    canvas = torch.zeros((3, *canvas_hw), dtype=torch.float32)
    content = torch.zeros(canvas_hw, dtype=torch.bool)
    canvas[:, top : top + h1, left : left + w1] = resized
    content[top : top + h1, left : left + w1] = True
    return canvas, content, padding


def letterbox_scalar(
    array: np.ndarray,
    canvas_hw: tuple[int, int] = (384, 512),
    *,
    mode: str = "nearest-exact",
    fill: float = 0.0,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    if array.ndim != 2:
        raise ValueError("letterbox_scalar requires [H,W]")
    (h1, w1), padding = target_geometry(tuple(array.shape), canvas_hw)
    top, _, left, _ = padding
    source = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
    kwargs: dict[str, Any] = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    resized = F.interpolate(source, size=(h1, w1), mode=mode, **kwargs)[0, 0]
    canvas = torch.full(canvas_hw, float(fill), dtype=torch.float32)
    canvas[top : top + h1, left : left + w1] = resized
    return canvas, padding


def unletterbox_scalar(
    canvas: torch.Tensor,
    padding: tuple[int, int, int, int],
    native_hw: tuple[int, int],
) -> torch.Tensor:
    if canvas.ndim != 2:
        raise ValueError("unletterbox_scalar requires [H,W]")
    top, bottom, left, right = padding
    h_end = canvas.shape[0] - bottom if bottom else canvas.shape[0]
    w_end = canvas.shape[1] - right if right else canvas.shape[1]
    content = canvas[top:h_end, left:w_end]
    return F.interpolate(
        content[None, None].float(), size=native_hw, mode="bilinear", align_corners=False,
        antialias=True,
    )[0, 0]


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def load_binary_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8).copy() > 0

