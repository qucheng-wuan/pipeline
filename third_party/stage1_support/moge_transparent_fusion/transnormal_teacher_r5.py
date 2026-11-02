"""Strict adapter and cache builder for the official TransNormal teacher.

R4 only borrowed a normal-loss shape.  R5 deliberately runs the released
TransNormal diffusion UNet with its released DINOv3 projector and uses the
resulting *full image* normal field as both a refiner input and supervision.
All heavyweight imports are delayed so audit/tests do not need diffusers.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from .io_utils import (
    atomic_json_dump,
    atomic_jsonl_dump,
    atomic_torch_save,
    canonical_json_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    target_geometry,
)

TEACHER_CACHE_SCHEMA = "moge_transparent_transnormal_teacher_cache_v1"
TEACHER_MANIFEST_SCHEMA = "moge_transparent_transnormal_teacher_manifest_v1"
COORDINATE_VERSION = "opencv_camera_xyz_front_facing_v1"
PLAN_SCHEMA = "moge_transparent_transnormal_teacher_r5_plan_v1"


@dataclass(frozen=True)
class TeacherAssetContract:
    model_root: Path
    source_root: Path
    dino_root: Path
    projector: Path
    revision: str
    source_commit: str
    processing_resolution: int
    timestep: int
    dtype: str
    canvas_hw: tuple[int, int]
    fingerprint: str


@dataclass(frozen=True)
class TeacherSource:
    identity: str
    dataset: str
    split: str
    sample_id: str
    rgb_sha256: str
    rgb_path: str | None = None
    group_id: str | None = None
    view_id: str | None = None
    record: Any | None = None
    asset: Any | None = None


def deterministic_teacher_seed(sample_id: str) -> int:
    digest = hashlib.sha256(f"transnormal-r5-v1|{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def transnormal_to_opencv(normal_01: Tensor, *, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Convert released TransNormal output to directed OpenCV camera normals.

    TransNormal encodes +X left, +Y up and +Z toward the viewer.  OpenCV uses
    the opposite three directions for the front-facing normal convention used
    by this project, so this is an overall negation, not a channel shuffle.
    Confidence records how close the decoded vector was to a unit vector before
    normalization; invalid/near-zero vectors get zero confidence.
    """

    if normal_01.ndim not in (3, 4) or normal_01.shape[-3] != 3:
        raise ValueError("normal_01 must be [3,H,W] or [B,3,H,W]")
    work = normal_01.float()
    finite = torch.isfinite(work).all(dim=-3, keepdim=True)
    decoded = -(2.0 * torch.where(finite, work, torch.full_like(work, 0.5)) - 1.0)
    magnitude = torch.linalg.vector_norm(decoded, dim=-3, keepdim=True)
    valid = finite & torch.isfinite(magnitude) & (magnitude > eps)
    normal = torch.where(valid, decoded / magnitude.clamp_min(eps), torch.zeros_like(decoded))
    confidence = torch.where(
        valid,
        (1.0 - (magnitude - 1.0).abs()).clamp(0.0, 1.0),
        torch.zeros_like(magnitude),
    )
    return normal, confidence


def letterbox_teacher_normal(
    normal_cv: Tensor,
    confidence: Tensor,
    canvas_hw: tuple[int, int],
) -> tuple[Tensor, Tensor, tuple[int, int, int, int]]:
    if normal_cv.ndim != 3 or normal_cv.shape[0] != 3:
        raise ValueError("normal_cv must be [3,H,W]")
    if confidence.shape != (1, *normal_cv.shape[-2:]):
        raise ValueError("confidence must be [1,H,W]")
    resized_hw, padding = target_geometry(tuple(normal_cv.shape[-2:]), canvas_hw)
    top, _, left, _ = padding
    normal = F.interpolate(
        normal_cv[None].float(), resized_hw, mode="bilinear", align_corners=False,
        antialias=True,
    )[0]
    conf = F.interpolate(
        confidence[None].float(), resized_hw, mode="bilinear", align_corners=False,
        antialias=True,
    )[0].clamp(0.0, 1.0)
    magnitude = torch.linalg.vector_norm(normal, dim=0, keepdim=True)
    valid = torch.isfinite(normal).all(dim=0, keepdim=True) & (magnitude > 1e-6)
    normal = torch.where(valid, normal / magnitude.clamp_min(1e-6), torch.zeros_like(normal))
    conf = torch.where(valid, conf, torch.zeros_like(conf))
    normal_canvas = torch.zeros((3, *canvas_hw), dtype=torch.float32)
    confidence_canvas = torch.zeros((1, *canvas_hw), dtype=torch.float32)
    h1, w1 = resized_hw
    normal_canvas[:, top : top + h1, left : left + w1] = normal
    confidence_canvas[:, top : top + h1, left : left + w1] = conf
    return normal_canvas, confidence_canvas, padding


def horizontal_flip_teacher(normal_cv: Tensor) -> Tensor:
    if normal_cv.shape[-3] != 3:
        raise ValueError("normal tensor must have a three-channel axis")
    result = normal_cv.flip(-1).clone()
    result.select(-3, 0).neg_()
    return result


def _verify_file(root: Path, entry: Mapping[str, Any], label: str) -> Path:
    path = (root / str(entry["path"])).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not a regular non-symlink file: {path}")
    if path.stat().st_size != int(entry["bytes"]):
        raise RuntimeError(f"{label} byte size mismatch: {path}")
    if sha256_file(path) != str(entry["sha256"]):
        raise RuntimeError(f"{label} SHA256 mismatch: {path}")
    return path


def validate_teacher_assets(plan_path: str | Path) -> tuple[dict[str, Any], TeacherAssetContract]:
    plan_file = Path(plan_path).expanduser().resolve(strict=True)
    plan = read_json(plan_file)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"unexpected R5 plan schema: {plan.get('schema_version')}")
    assets = plan["assets"]
    teacher = plan["teacher"]
    if teacher.get("required") is not True:
        raise ValueError("formal R5 requires the official TransNormal teacher")
    model_root = Path(assets["transnormal_model"]).resolve(strict=True)
    source_root = Path(assets["transnormal_source"]).resolve(strict=True)
    dino_root = Path(assets["dinov3_model"]).resolve(strict=True)
    if not all(path.is_dir() for path in (model_root, source_root, dino_root)):
        raise NotADirectoryError("TransNormal/DINO source roots must be directories")
    unet = _verify_file(model_root, assets["transnormal_unet"], "TransNormal UNet")
    vae = _verify_file(model_root, assets["transnormal_vae"], "TransNormal VAE")
    projector = _verify_file(
        model_root, assets["transnormal_projector"], "TransNormal DINO projector"
    )
    dino = _verify_file(dino_root, assets["dinov3_weights"], "DINOv3 weights")
    for relative in (
        "model_index.json",
        "unet/config.json",
        "vae/config.json",
        "scheduler/scheduler_config.json",
    ):
        if not (model_root / relative).is_file():
            raise RuntimeError(f"incomplete TransNormal snapshot: missing {relative}")
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != assets["transnormal_source_commit"]:
        raise RuntimeError(f"TransNormal source commit {commit} is not pinned")
    contract_value = {
        "schema_version": TEACHER_CACHE_SCHEMA,
        "revision": assets["transnormal_revision"],
        "source_commit": commit,
        "unet_sha256": sha256_file(unet),
        "vae_sha256": sha256_file(vae),
        "projector_sha256": sha256_file(projector),
        "dino_revision": assets["dinov3_revision"],
        "dino_sha256": sha256_file(dino),
        "processing_resolution": int(teacher["processing_resolution"]),
        "timestep": int(teacher["timestep"]),
        "dtype": str(teacher["dtype"]),
        "canvas_hw": [int(value) for value in teacher["canvas_hw"]],
        "coordinate_version": COORDINATE_VERSION,
    }
    return plan, TeacherAssetContract(
        model_root=model_root,
        source_root=source_root,
        dino_root=dino_root,
        projector=projector,
        revision=str(assets["transnormal_revision"]),
        source_commit=commit,
        processing_resolution=int(teacher["processing_resolution"]),
        timestep=int(teacher["timestep"]),
        dtype=str(teacher["dtype"]),
        canvas_hw=tuple(int(value) for value in teacher["canvas_hw"]),
        fingerprint=canonical_json_sha256(contract_value),
    )


def load_official_teacher(plan_path: str | Path, device: str | torch.device):
    """Load the exact released teacher; never permits CLIP/random-projector fallback."""

    _, contract = validate_teacher_assets(plan_path)
    if contract.dtype != "bfloat16":
        raise ValueError("the audited R5 teacher dtype is bfloat16")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source = str(contract.source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    from transnormal import TransNormalPipeline, create_dino_encoder

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("formal TransNormal teacher inference requires CUDA")
    dino_encoder = create_dino_encoder(
        model_name="dinov3_vith16plus",
        weights_path=str(contract.dino_root),
        projector_path=str(contract.projector),
        device=target,
        dtype=torch.bfloat16,
        freeze_encoder=True,
    )
    pipeline = TransNormalPipeline.from_pretrained(
        str(contract.model_root),
        dino_encoder=dino_encoder,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(target)
    if not bool(getattr(pipeline, "_use_dino_for_cross_attention", False)):
        raise RuntimeError("TransNormal loaded without DINO cross-attention")
    if getattr(pipeline, "dino_encoder", None) is None:
        raise RuntimeError("TransNormal DINO encoder is missing")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, contract


@torch.inference_mode()
def infer_teacher_normal(
    pipeline: Any,
    contract: TeacherAssetContract,
    rgb_u8: np.ndarray,
    *,
    sample_id: str,
) -> dict[str, Any]:
    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("teacher RGB must be uint8 [H,W,3]")
    seed = deterministic_teacher_seed(sample_id)
    device = pipeline._execution_device
    devices = [device.index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        output = pipeline(
            image=rgb_u8,
            timestep=contract.timestep,
            processing_res=contract.processing_resolution,
            match_input_res=True,
            resample_method="bilinear",
            output_type="pt",
        )
    if not isinstance(output, Tensor) or output.shape != (1, 3, *rgb_u8.shape[:2]):
        raise RuntimeError(f"unexpected TransNormal output shape: {getattr(output, 'shape', None)}")
    normal, confidence = transnormal_to_opencv(output[0].float().cpu())
    normal_canvas, confidence_canvas, padding = letterbox_teacher_normal(
        normal, confidence, contract.canvas_hw
    )
    if not bool(torch.isfinite(normal_canvas).all() and torch.isfinite(confidence_canvas).all()):
        raise RuntimeError("TransNormal teacher produced non-finite output")
    return {
        "normal_cv": normal_canvas,
        "confidence": confidence_canvas,
        "padding_tblr": padding,
        "native_hw": tuple(int(value) for value in rgb_u8.shape[:2]),
        "seed": seed,
    }


def _cleargrasp_sources(roster_path: Path) -> list[TeacherSource]:
    result = []
    for row in read_jsonl(roster_path):
        if row.get("role") not in {"train", "validation", "evaluation"}:
            continue
        identity = f"cleargrasp/{row['role']}/{row['cache_id']}"
        result.append(
            TeacherSource(
                identity=identity,
                dataset="cleargrasp",
                split=str(row["role"]),
                sample_id=str(row["sample_id"]),
                rgb_sha256=str(row["rgb_sha256"]),
                rgb_path=str(row["rgb_path"]),
            )
        )
    return result


def _transnormal_sources(manifest_path: Path) -> list[TeacherSource]:
    result = []
    for record in load_intervention_manifest(manifest_path):
        for view in record.views:
            identity = f"transnormal/{record.split}/{record.group_id}/{view.view_id}"
            result.append(
                TeacherSource(
                    identity=identity,
                    dataset="transnormal",
                    split=record.split,
                    sample_id=f"{record.group_id}/{view.view_id}",
                    group_id=record.group_id,
                    view_id=view.view_id,
                    rgb_sha256=view.rgb.sha256,
                    record=record,
                    asset=view.rgb,
                )
            )
    return result


def teacher_sources(
    dataset: str, source_manifest: str | Path
) -> list[TeacherSource]:
    path = Path(source_manifest).expanduser().resolve(strict=True)
    if dataset == "cleargrasp":
        return _cleargrasp_sources(path)
    if dataset == "transnormal":
        return _transnormal_sources(path)
    raise ValueError("dataset must be cleargrasp or transnormal")


def _load_source_rgb(source: TeacherSource) -> np.ndarray:
    if source.dataset == "cleargrasp":
        path = Path(str(source.rgb_path)).resolve(strict=True)
        if sha256_file(path) != source.rgb_sha256:
            raise RuntimeError(f"RGB SHA256 mismatch: {source.identity}")
        payload = path.read_bytes()
    else:
        payload = _read_asset(source.record, source.asset)
        if hashlib.sha256(payload).hexdigest() != source.rgb_sha256:
            raise RuntimeError(f"archive RGB SHA256 mismatch: {source.identity}")
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _cache_path(cache_root: Path, source: TeacherSource) -> Path:
    token = hashlib.sha256(source.identity.encode()).hexdigest()[:24]
    return cache_root / source.dataset / source.split / f"{token}.pt"


def _cache_manifest_row(
    path: Path,
    source: TeacherSource,
    contract: TeacherAssetContract,
) -> dict[str, Any]:
    return {
        "schema_version": TEACHER_MANIFEST_SCHEMA,
        "identity": source.identity,
        "dataset": source.dataset,
        "split": source.split,
        "sample_id": source.sample_id,
        "group_id": source.group_id,
        "view_id": source.view_id,
        "rgb_sha256": source.rgb_sha256,
        "teacher_fingerprint": contract.fingerprint,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_reusable_cache(
    path: Path,
    source: TeacherSource,
    contract: TeacherAssetContract,
) -> dict[str, Any] | None:
    """Return a manifest row for an atomic completed payload, else fail closed."""

    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"unsafe pre-existing teacher cache path: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"pre-existing teacher cache is not a mapping: {path}")
    expected = {
        "schema_version": TEACHER_CACHE_SCHEMA,
        "identity": source.identity,
        "dataset": source.dataset,
        "split": source.split,
        "sample_id": source.sample_id,
        "group_id": source.group_id,
        "view_id": source.view_id,
        "rgb_sha256": source.rgb_sha256,
        "teacher_fingerprint": contract.fingerprint,
        "teacher_revision": contract.revision,
        "teacher_source_commit": contract.source_commit,
        "coordinate_version": COORDINATE_VERSION,
        "seed": deterministic_teacher_seed(source.sample_id),
    }
    mismatch = [key for key, item in expected.items() if value.get(key) != item]
    normal = value.get("normal_cv")
    confidence = value.get("confidence")
    tensor_valid = (
        isinstance(normal, Tensor)
        and isinstance(confidence, Tensor)
        and normal.shape == (3, *contract.canvas_hw)
        and confidence.shape == (1, *contract.canvas_hw)
        and bool(torch.isfinite(normal).all())
        and bool(torch.isfinite(confidence).all())
        and bool(((confidence >= 0) & (confidence <= 1)).all())
    )
    if mismatch or not tensor_valid:
        raise RuntimeError(
            f"pre-existing teacher cache does not match the locked contract: "
            f"{path}; fields={mismatch}"
        )
    return _cache_manifest_row(path, source, contract)


def build_teacher_cache_worker(
    plan_path: str | Path,
    *,
    dataset: str,
    source_manifest: str | Path,
    rank: int,
    world_size: int,
    device: str,
    limit: int | None = None,
) -> Path:
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise ValueError("invalid rank/world_size")
    plan, _ = validate_teacher_assets(plan_path)
    pipeline, contract = load_official_teacher(plan_path, device)
    cache_root = Path(plan["teacher"]["cache_root"]).resolve()
    sources = teacher_sources(dataset, source_manifest)
    selected = sources[rank::world_size]
    if limit is not None:
        selected = selected[: int(limit)]
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(selected, 1):
        path = _cache_path(cache_root, source)
        reusable = _validate_reusable_cache(path, source, contract)
        if reusable is not None:
            rows.append(reusable)
            print(
                f"teacher {dataset} rank={rank} {index}/{len(selected)} "
                f"{source.identity} reused",
                flush=True,
            )
            continue
        rgb = _load_source_rgb(source)
        result = infer_teacher_normal(
            pipeline, contract, rgb, sample_id=source.sample_id
        )
        payload = {
            "schema_version": TEACHER_CACHE_SCHEMA,
            "identity": source.identity,
            "dataset": source.dataset,
            "split": source.split,
            "sample_id": source.sample_id,
            "group_id": source.group_id,
            "view_id": source.view_id,
            "rgb_sha256": source.rgb_sha256,
            "teacher_fingerprint": contract.fingerprint,
            "teacher_revision": contract.revision,
            "teacher_source_commit": contract.source_commit,
            "coordinate_version": COORDINATE_VERSION,
            "seed": result["seed"],
            "native_hw": torch.tensor(result["native_hw"], dtype=torch.int64),
            "padding_tblr": torch.tensor(result["padding_tblr"], dtype=torch.int64),
            "normal_cv": result["normal_cv"].to(torch.float16).contiguous(),
            "confidence": result["confidence"].to(torch.float16).contiguous(),
        }
        atomic_torch_save(payload, path)
        rows.append(_cache_manifest_row(path, source, contract))
        print(f"teacher {dataset} rank={rank} {index}/{len(selected)} {source.identity}", flush=True)
    worker = cache_root.parent / "workers" / f"{dataset}.rank_{rank:02d}.jsonl"
    atomic_jsonl_dump(rows, worker)
    atomic_json_dump(
        {
            "schema_version": TEACHER_MANIFEST_SCHEMA,
            "status": "worker_complete",
            "dataset": dataset,
            "rank": rank,
            "world_size": world_size,
            "samples": len(rows),
            "teacher_fingerprint": contract.fingerprint,
            "manifest": str(worker),
            "manifest_sha256": sha256_file(worker),
        },
        worker.with_suffix(".complete.json"),
    )
    return worker


def merge_teacher_cache(
    plan_path: str | Path,
    *,
    datasets: Iterable[tuple[str, str | Path]],
    world_size: int,
) -> Path:
    plan, contract = validate_teacher_assets(plan_path)
    cache_root = Path(plan["teacher"]["cache_root"]).resolve()
    expected: dict[str, TeacherSource] = {}
    rows: list[dict[str, Any]] = []
    for dataset, source_manifest in datasets:
        for source in teacher_sources(dataset, source_manifest):
            if source.identity in expected:
                raise ValueError(f"duplicate teacher source identity: {source.identity}")
            expected[source.identity] = source
        for rank in range(world_size):
            worker = cache_root.parent / "workers" / f"{dataset}.rank_{rank:02d}.jsonl"
            complete = read_json(worker.with_suffix(".complete.json"))
            if complete.get("status") != "worker_complete" or complete.get(
                "teacher_fingerprint"
            ) != contract.fingerprint:
                raise ValueError(f"invalid teacher worker completion: {worker}")
            if sha256_file(worker) != complete.get("manifest_sha256"):
                raise ValueError(f"teacher worker manifest changed: {worker}")
            rows.extend(read_jsonl(worker))
    by_identity = {row.get("identity"): row for row in rows}
    if len(by_identity) != len(rows) or set(by_identity) != set(expected):
        missing = sorted(set(expected) - set(by_identity))[:5]
        extra = sorted(set(by_identity) - set(expected))[:5]
        raise ValueError(f"teacher cache coverage mismatch; missing={missing}, extra={extra}")
    ordered = []
    for identity in sorted(expected):
        source = expected[identity]
        row = by_identity[identity]
        path = Path(row["path"]).resolve(strict=True)
        if (
            row.get("rgb_sha256") != source.rgb_sha256
            or row.get("teacher_fingerprint") != contract.fingerprint
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"teacher cache row failed validation: {identity}")
        ordered.append(row)
    manifest = Path(plan["teacher"]["manifest"]).resolve()
    atomic_jsonl_dump(ordered, manifest)
    atomic_json_dump(
        {
            "schema_version": TEACHER_MANIFEST_SCHEMA,
            "status": "complete",
            "samples": len(ordered),
            "dataset_counts": {
                name: sum(row["dataset"] == name for row in ordered)
                for name in sorted({row["dataset"] for row in ordered})
            },
            "teacher_fingerprint": contract.fingerprint,
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "exact_coverage": True,
        },
        manifest.with_suffix(".complete.json"),
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("cache-worker")
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--dataset", choices=("cleargrasp", "transnormal"), required=True)
    worker.add_argument("--source-manifest", type=Path, required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--world-size", type=int, required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--limit", type=int)
    merge = sub.add_parser("merge")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--cleargrasp-roster", type=Path, required=True)
    merge.add_argument("--transnormal-manifest", type=Path, required=True)
    merge.add_argument("--world-size", type=int, required=True)
    audit = sub.add_parser("audit-assets")
    audit.add_argument("--plan", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit-assets":
        _, contract = validate_teacher_assets(args.plan)
        print(contract)
    elif args.command == "cache-worker":
        build_teacher_cache_worker(
            args.plan,
            dataset=args.dataset,
            source_manifest=args.source_manifest,
            rank=args.rank,
            world_size=args.world_size,
            device=args.device,
            limit=args.limit,
        )
    else:
        merge_teacher_cache(
            args.plan,
            datasets=(
                ("cleargrasp", args.cleargrasp_roster),
                ("transnormal", args.transnormal_manifest),
            ),
            world_size=args.world_size,
        )


if __name__ == "__main__":
    main()

