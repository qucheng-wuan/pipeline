"""Label-free, sharded TOM-Net probability inference over the full roster.

Workers read RGB files only.  Ground-truth mask and depth paths are never
looked up, derived, or opened.  Eight independent workers shard the immutable
roster by ``roster_index % world_size``; a CPU merge then verifies exact file
coverage and seals a deterministic manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .io_utils import (
    atomic_json_dump,
    atomic_jsonl_dump,
    atomic_torch_save,
    canonical_json_sha256,
    letterbox_rgb_raw,
    load_rgb,
    read_json,
    read_jsonl,
    sha256_file,
)
from .tomnet import (
    CONVERTED_CHECKPOINT_FORMAT,
    OBJECT_MASK_CHANNEL,
    TomNetCoarseNet,
    load_tomnet_coarsenet,
)


CANVAS_HW = (384, 512)
DEFAULT_WORLD_SIZE = 8
EXPECTED_ROLE_COUNTS = {"train": 44_953, "validation": 500, "evaluation": 1_226}
EXPECTED_SAMPLES = sum(EXPECTED_ROLE_COUNTS.values())

TRAINED_CHECKPOINT_SCHEMA = "tomnet_cleargrasp_mask_checkpoint_v1"
CALIBRATION_SCHEMA = "tomnet_cleargrasp_mask_calibration_v1"
PROBABILITY_SCHEMA = "tomnet_calibrated_probability_canvas_v1"
WORKER_SCHEMA = "tomnet_probability_worker_v1"
MANIFEST_SCHEMA = "tomnet_probability_manifest_v1"
INFERENCE_SOURCE_CONTRACT_SCHEMA = "tomnet_inference_source_contract_v1"


def tomnet_inference_source_contract() -> dict[str, Any]:
    """Return the content-addressed implementation contract for inference.

    A TOM probability is only reusable when the code that defines inference,
    preprocessing/IO, and the model graph is byte-identical. Keep paths out of
    the canonical value so the seal depends on source content rather than the
    checkout location.
    """

    module_root = Path(__file__).resolve(strict=True).parent
    contract = {
        "schema_version": INFERENCE_SOURCE_CONTRACT_SCHEMA,
        "tomnet_infer_module_sha256": sha256_file(module_root / "tomnet_infer.py"),
        "io_utils_module_sha256": sha256_file(module_root / "io_utils.py"),
        "tomnet_module_sha256": sha256_file(module_root / "tomnet.py"),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def _validate_inference_source_contract(
    value: Any, *, label: str
) -> dict[str, Any]:
    """Fail closed unless a recorded source contract matches this checkout."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    expected = tomnet_inference_source_contract()
    if dict(value) != expected:
        mismatched = sorted(
            key
            for key in set(value) | set(expected)
            if value.get(key) != expected.get(key)
        )
        raise ValueError(f"{label} mismatch: {mismatched}")
    return expected


def _require_file(value: str | Path, label: str) -> Path:
    candidate = Path(value).expanduser()
    try:
        path = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {candidate}") from None
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _sealed_worker_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove restart-only state from the content-addressed worker manifest."""

    return [
        {key: value for key, value in record.items() if key != "status"}
        for record in records
    ]


def _sealed_worker_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Keep immutable proof fields; leave telemetry in the rank completion file."""

    volatile = {
        "created",
        "reused",
        "device",
        "device_name",
        "peak_cuda_bytes",
        "wall_seconds",
    }
    return {key: value for key, value in summary.items() if key not in volatile}


def _torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch.
        return torch.load(path, map_location=map_location)


def temperature_scaled_object_probability(
    mask_logits: Tensor, temperature: float
) -> Tensor:
    """Return calibrated channel-1 probability from two-class mask logits.

    TOM-Net's ``object_probability`` property uses the uncalibrated softmax.
    Temperature scaling must therefore operate on the binary logit difference
    ``mask_logits[:, 1] - mask_logits[:, 0]`` before applying sigmoid.
    """

    if mask_logits.ndim != 4 or mask_logits.shape[1] != 2:
        raise ValueError("TOM-Net final mask logits must have shape [B,2,H,W]")
    value = float(temperature)
    if not math.isfinite(value) or not 0.05 <= value <= 20.0:
        raise ValueError("calibration temperature must lie in [0.05,20]")
    difference = (
        mask_logits[:, OBJECT_MASK_CHANNEL : OBJECT_MASK_CHANNEL + 1].float()
        - mask_logits[:, 0:1].float()
    )
    probability = torch.sigmoid(difference / value)
    if not bool(torch.isfinite(probability).all()):
        raise FloatingPointError("calibrated TOM-Net probability is non-finite")
    return probability


def _autocast_context(device: torch.device, amp_dtype: str):
    """Match the precision used when the checkpoint calibration was fitted."""

    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    if amp_dtype not in {"fp16", "bf16"}:
        raise ValueError(f"unsupported TOM-Net inference amp dtype: {amp_dtype!r}")
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def load_inference_model(
    checkpoint_path: str | Path, device: str | torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained mask checkpoint or a converted official checkpoint.

    ``load_tomnet_coarsenet`` natively handles converted/Torch7 checkpoints.
    The ClearGrasp trainer instead writes a complete state dict under
    ``tomnet_cleargrasp_mask_checkpoint_v1``; that explicit compatibility path
    is implemented here and uses strict state loading.
    """

    checkpoint = _require_file(checkpoint_path, "TOM-Net checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint)
    payload = _torch_load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("TOM-Net checkpoint root must be a mapping")

    embedded_calibration: Mapping[str, Any] | None = None
    if payload.get("schema_version") == TRAINED_CHECKPOINT_SCHEMA:
        if payload.get("mask_class_index") != OBJECT_MASK_CHANNEL:
            raise ValueError("trained checkpoint mask_class_index must be 1")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("trained TOM-Net checkpoint lacks a state_dict")
        model: nn.Module = TomNetCoarseNet()
        model.load_state_dict(state_dict, strict=True)
        candidate = payload.get("calibration")
        if isinstance(candidate, Mapping):
            embedded_calibration = candidate
        training_arguments = payload.get("training_arguments")
        inference_amp_dtype = (
            str(training_arguments.get("amp_dtype", "none"))
            if isinstance(training_arguments, Mapping)
            else "none"
        )
        if inference_amp_dtype not in {"none", "fp16", "bf16"}:
            raise ValueError(
                "trained checkpoint has an unsupported training amp dtype: "
                f"{inference_amp_dtype!r}"
            )
        metadata = {
            "schema_version": TRAINED_CHECKPOINT_SCHEMA,
            "epoch": int(payload["epoch"]),
            "inference_amp_dtype": inference_amp_dtype,
            "embedded_calibration": (
                dict(embedded_calibration) if embedded_calibration is not None else None
            ),
        }
    elif payload.get("format") == CONVERTED_CHECKPOINT_FORMAT:
        model = load_tomnet_coarsenet(checkpoint, map_location="cpu")
        metadata = {
            "schema_version": CONVERTED_CHECKPOINT_FORMAT,
            "epoch": None,
            "inference_amp_dtype": "none",
            "embedded_calibration": None,
        }
    else:
        raise ValueError(
            "unsupported TOM-Net checkpoint schema/format: "
            f"schema_version={payload.get('schema_version')!r}, "
            f"format={payload.get('format')!r}"
        )

    model = model.to(device).eval().requires_grad_(False)
    metadata.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    return model, metadata


def load_calibration(
    calibration_path: str | Path,
    checkpoint_path: str | Path,
    *,
    embedded_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load calibration and strictly bind it to the supplied checkpoint."""

    calibration_file = _require_file(calibration_path, "TOM-Net calibration")
    checkpoint = _require_file(checkpoint_path, "TOM-Net checkpoint")
    value = read_json(calibration_file)
    if not isinstance(value, dict):
        raise ValueError("TOM-Net calibration must be a JSON object")
    if value.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(
            f"unsupported TOM-Net calibration schema: {value.get('schema_version')!r}"
        )
    if value.get("status") != "complete":
        raise ValueError("TOM-Net calibration status must be complete")
    declared_checkpoint = _require_file(value["checkpoint"], "calibration checkpoint")
    if declared_checkpoint != checkpoint:
        raise ValueError(
            f"calibration checkpoint mismatch: {declared_checkpoint} != {checkpoint}"
        )
    checkpoint_sha256 = sha256_file(checkpoint)
    if value.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("calibration checkpoint SHA256 mismatch")

    temperature = float(value["temperature"])
    # Use the production probability function as the single bounds/finite check.
    temperature_scaled_object_probability(torch.zeros(1, 2, 1, 1), temperature)
    gate_threshold = float(value["gate_threshold"])
    if not math.isfinite(gate_threshold) or not 0.0 <= gate_threshold < 1.0:
        raise ValueError("gate_threshold must be finite and lie in [0,1)")

    if embedded_calibration is not None:
        embedded_temperature = float(embedded_calibration["temperature"])
        embedded_threshold = float(embedded_calibration["threshold"])
        if not math.isclose(
            temperature, embedded_temperature, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("external and checkpoint-embedded temperatures differ")
        if not math.isclose(
            gate_threshold, embedded_threshold, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("external gate threshold and checkpoint threshold differ")

    return {
        **value,
        "path": str(calibration_file),
        "sha256": sha256_file(calibration_file),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "temperature": temperature,
        "gate_threshold": gate_threshold,
    }


def _roots(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = (
        Path(config["paths"]["artifact_root"]).expanduser().resolve() / "tomnet_masks"
    )
    return root, root / "probabilities", root / "workers"


def _validate_roster(rows: list[dict[str, Any]]) -> None:
    cache_ids = [str(row["cache_id"]) for row in rows]
    if len(cache_ids) != len(set(cache_ids)):
        raise ValueError("full roster cache IDs are not unique")
    allowed_roles = set(EXPECTED_ROLE_COUNTS)
    unknown_roles = sorted({str(row.get("role")) for row in rows} - allowed_roles)
    if unknown_roles:
        raise ValueError(f"full roster contains unknown roles: {unknown_roles}")
    for index, row in enumerate(rows):
        for key in (
            "cache_id",
            "sample_id",
            "role",
            "split",
            "rgb_path",
            "rgb_sha256",
        ):
            if key not in row:
                raise ValueError(f"roster row {index} lacks {key}")


def _validate_cached_probability(
    path: Path,
    row: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    calibration_sha256: str,
    inference_source_contract_sha256: str,
) -> dict[str, Any]:
    payload = _torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"probability cache is not a mapping: {path}")
    expected = {
        "schema_version": PROBABILITY_SCHEMA,
        "cache_id": row["cache_id"],
        "sample_id": row["sample_id"],
        "role": row["role"],
        "rgb_sha256": row["rgb_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
        "inference_source_contract_sha256": inference_source_contract_sha256,
        "contains_ground_truth": False,
        "ground_truth_mask_access": False,
        "ground_truth_depth_access": False,
    }
    mismatched = [
        key
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if mismatched:
        raise ValueError(f"incompatible probability cache {path}: {mismatched}")
    probability = payload.get("probability")
    if not isinstance(probability, Tensor) or tuple(probability.shape) != CANVAS_HW:
        raise ValueError(f"invalid probability canvas in {path}")
    if probability.dtype != torch.float16:
        raise ValueError(f"probability canvas must be float16: {path}")
    if not bool(torch.isfinite(probability).all()) or not bool(
        ((probability >= 0) & (probability <= 1)).all()
    ):
        raise ValueError(f"invalid probability values in {path}")
    padding = payload.get("padding_tblr")
    native_hw = payload.get("native_hw")
    if not isinstance(padding, Tensor) or tuple(padding.shape) != (4,):
        raise ValueError(f"invalid padding tensor in {path}")
    if not isinstance(native_hw, Tensor) or tuple(native_hw.shape) != (2,):
        raise ValueError(f"invalid native_hw tensor in {path}")
    return payload


@torch.inference_mode()
def infer_worker(
    config_path: str | Path,
    roster_path: str | Path,
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    *,
    rank: int,
    world_size: int = DEFAULT_WORLD_SIZE,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run one deterministic RGB-only roster shard."""

    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(
            "rank must lie in [0, world_size) and world_size must be positive"
        )
    config_file = _require_file(config_path, "configuration")
    roster_file = _require_file(roster_path, "full roster")
    checkpoint = _require_file(checkpoint_path, "TOM-Net checkpoint")
    calibration_file = _require_file(calibration_path, "TOM-Net calibration")
    config = read_json(config_file)
    if tuple(int(value) for value in config["canvas_hw"]) != CANVAS_HW:
        raise ValueError(f"TOM-Net canvas must be exactly {CANVAS_HW}")
    rows = read_jsonl(roster_file)
    _validate_roster(rows)
    inference_source_contract = tomnet_inference_source_contract()
    inference_source_contract_sha256 = inference_source_contract["contract_sha256"]
    selected = [
        (index, row) for index, row in enumerate(rows) if index % world_size == rank
    ]

    device = torch.device(device_name or f"cuda:{rank}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_float32_matmul_precision("high")
    model, checkpoint_metadata = load_inference_model(checkpoint, device)
    inference_amp_dtype = str(
        checkpoint_metadata.get("inference_amp_dtype", "none")
    )
    # Validate this before processing the first image, including on CPU where
    # autocast itself would otherwise be a no-op.
    if inference_amp_dtype not in {"none", "fp16", "bf16"}:
        raise ValueError(
            f"unsupported TOM-Net inference amp dtype: {inference_amp_dtype!r}"
        )
    calibration = load_calibration(
        calibration_file,
        checkpoint,
        embedded_calibration=checkpoint_metadata.get("embedded_calibration"),
    )
    temperature = float(calibration["temperature"])
    root, probability_root, worker_root = _roots(config)
    del root
    records: list[dict[str, Any]] = []
    started = time.time()

    for local_index, (roster_index, row) in enumerate(selected, start=1):
        tick = time.perf_counter()
        destination = probability_root / str(row["role"]) / f"{row['cache_id']}.pt"
        if destination.is_file():
            _validate_cached_probability(
                destination,
                row,
                checkpoint_sha256=checkpoint_metadata["checkpoint_sha256"],
                calibration_sha256=calibration["sha256"],
                inference_source_contract_sha256=(
                    inference_source_contract_sha256
                ),
            )
            status = "reused"
        else:
            # This is the only source-data read in the inference loop.
            rgb = load_rgb(
                _require_file(row["rgb_path"], f"RGB for {row['sample_id']}")
            )
            native_hw = tuple(int(value) for value in rgb.shape[:2])
            canvas, content, padding = letterbox_rgb_raw(rgb, CANVAS_HW)
            context = _autocast_context(device, inference_amp_dtype)
            with context:
                output = model(canvas.unsqueeze(0).to(device, non_blocking=True))
            final = getattr(output, "final", None)
            mask_logits = getattr(final, "mask_logits", None)
            if not isinstance(mask_logits, Tensor):
                raise TypeError(
                    "TomNetCoarseNet output.final.mask_logits is unavailable"
                )
            if tuple(mask_logits.shape) != (1, 2, *CANVAS_HW):
                raise ValueError(
                    f"unexpected final mask-logit shape for {row['sample_id']}: "
                    f"{tuple(mask_logits.shape)}"
                )
            probability = temperature_scaled_object_probability(
                mask_logits, temperature
            )[0, 0].cpu()
            # Artificial letterbox pixels are never valid scene evidence.
            probability = probability.masked_fill(~content, 0.0)
            payload = {
                "schema_version": PROBABILITY_SCHEMA,
                "cache_id": row["cache_id"],
                "sample_id": row["sample_id"],
                "role": row["role"],
                "rgb_sha256": row["rgb_sha256"],
                "probability": probability.to(torch.float16).contiguous(),
                "padding_tblr": torch.tensor(padding, dtype=torch.int16),
                "native_hw": torch.tensor(native_hw, dtype=torch.int32),
                "temperature": temperature,
                "checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
                "calibration_sha256": calibration["sha256"],
                "inference_source_contract_sha256": (
                    inference_source_contract_sha256
                ),
                "contains_ground_truth": False,
                "ground_truth_mask_access": False,
                "ground_truth_depth_access": False,
            }
            atomic_torch_save(payload, destination)
            status = "created"

        probability_sha256 = sha256_file(destination)
        records.append(
            {
                "roster_index": roster_index,
                "rank": rank,
                "cache_id": row["cache_id"],
                "sample_id": row["sample_id"],
                "role": row["role"],
                "split": row["split"],
                "probability_path": str(destination.resolve()),
                "probability_sha256": probability_sha256,
                "inference_source_contract_sha256": (
                    inference_source_contract_sha256
                ),
                "bytes": destination.stat().st_size,
                "status": status,
                "contains_ground_truth": False,
            }
        )
        if local_index == 1 or local_index % 100 == 0 or local_index == len(selected):
            print(
                f"tomnet-mask rank={rank} [{local_index}/{len(selected)}] "
                f"{row['cache_id']} {time.perf_counter() - tick:.3f}s",
                flush=True,
            )

    worker_manifest = worker_root / f"rank_{rank:02d}.jsonl"
    _validate_inference_source_contract(
        inference_source_contract,
        label="TOM-Net inference source contract at worker completion",
    )
    atomic_jsonl_dump(_sealed_worker_records(records), worker_manifest)
    summary = {
        "schema_version": WORKER_SCHEMA,
        "status": "complete",
        "rank": rank,
        "world_size": world_size,
        "samples": len(records),
        "created": sum(row["status"] == "created" for row in records),
        "reused": sum(row["status"] == "reused" for row in records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "roster": str(roster_file),
        "roster_sha256": sha256_file(roster_file),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
        "checkpoint_schema": checkpoint_metadata["schema_version"],
        "inference_amp_dtype": inference_amp_dtype,
        "calibration": str(calibration_file),
        "calibration_sha256": calibration["sha256"],
        "inference_source_contract": inference_source_contract,
        "inference_source_contract_sha256": inference_source_contract_sha256,
        "temperature": temperature,
        "gate_threshold": calibration["gate_threshold"],
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "ground_truth_mask_access": False,
        "ground_truth_depth_access": False,
        "contains_ground_truth": False,
        "wall_seconds": time.time() - started,
    }
    atomic_json_dump(summary, worker_root / f"rank_{rank:02d}.complete.json")
    return summary


def _merge_tomnet_masks(
    config_path: str | Path,
    roster_path: str | Path,
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    *,
    world_size: int,
    expected_role_counts: Mapping[str, int],
) -> dict[str, Any]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    config_file = _require_file(config_path, "configuration")
    roster_file = _require_file(roster_path, "full roster")
    checkpoint = _require_file(checkpoint_path, "TOM-Net checkpoint")
    calibration_file = _require_file(calibration_path, "TOM-Net calibration")
    config = read_json(config_file)
    if tuple(int(value) for value in config["canvas_hw"]) != CANVAS_HW:
        raise ValueError(f"TOM-Net canvas must be exactly {CANVAS_HW}")
    rows = read_jsonl(roster_file)
    _validate_roster(rows)
    inference_source_contract = tomnet_inference_source_contract()
    inference_source_contract_sha256 = inference_source_contract["contract_sha256"]
    role_counts = Counter(str(row["role"]) for row in rows)
    if dict(role_counts) != dict(expected_role_counts):
        raise ValueError(
            f"full roster role counts mismatch: {dict(role_counts)} != "
            f"{dict(expected_role_counts)}"
        )
    expected_samples = sum(expected_role_counts.values())
    if len(rows) != expected_samples:
        raise ValueError(f"full roster must contain exactly {expected_samples} rows")

    calibration = load_calibration(calibration_file, checkpoint)
    checkpoint_sha256 = calibration["checkpoint_sha256"]
    roster_sha256 = sha256_file(roster_file)
    root, probability_root, worker_root = _roots(config)
    all_records: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    for rank in range(world_size):
        summary_path = _require_file(
            worker_root / f"rank_{rank:02d}.complete.json", f"worker {rank} completion"
        )
        manifest_path = _require_file(
            worker_root / f"rank_{rank:02d}.jsonl", f"worker {rank} manifest"
        )
        summary = read_json(summary_path)
        expected_summary = {
            "schema_version": WORKER_SCHEMA,
            "status": "complete",
            "rank": rank,
            "world_size": world_size,
            "roster": str(roster_file),
            "roster_sha256": roster_sha256,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "calibration": str(calibration_file),
            "calibration_sha256": calibration["sha256"],
            "inference_source_contract": inference_source_contract,
            "inference_source_contract_sha256": (
                inference_source_contract_sha256
            ),
            "ground_truth_mask_access": False,
            "ground_truth_depth_access": False,
            "contains_ground_truth": False,
        }
        mismatched = [
            key
            for key, expected_value in expected_summary.items()
            if summary.get(key) != expected_value
        ]
        if mismatched:
            raise ValueError(f"worker {rank} completion mismatch: {mismatched}")
        records = read_jsonl(manifest_path)
        expected_indices = list(range(rank, len(rows), world_size))
        observed_indices = [int(record["roster_index"]) for record in records]
        if observed_indices != expected_indices:
            raise ValueError(f"worker {rank} does not exactly cover its roster shard")
        if int(summary.get("samples", -1)) != len(records):
            raise ValueError(f"worker {rank} sample count differs from its manifest")
        for record, roster_index in zip(records, expected_indices, strict=True):
            source = rows[roster_index]
            if "status" in record:
                raise ValueError(
                    f"worker {rank} manifest contains runtime-only status at "
                    f"{roster_index}"
                )
            for key in ("cache_id", "sample_id", "role", "split"):
                if record.get(key) != source.get(key):
                    raise ValueError(
                        f"worker {rank} record identity mismatch at {roster_index}"
                    )
            if record.get("contains_ground_truth") is not False:
                raise ValueError(f"worker {rank} manifest claims ground truth")
            if record.get("rank") != rank:
                raise ValueError(
                    f"worker {rank} record rank mismatch at {roster_index}"
                )
            if (
                record.get("inference_source_contract_sha256")
                != inference_source_contract_sha256
            ):
                raise ValueError(
                    f"worker {rank} source contract mismatch at {roster_index}"
                )
            expected_path = (
                probability_root / str(source["role"]) / f"{source['cache_id']}.pt"
            ).resolve()
            path = _require_file(record["probability_path"], "probability cache")
            if path != expected_path:
                raise ValueError(f"probability cache escaped canonical root: {path}")
            if sha256_file(path) != record.get("probability_sha256"):
                raise ValueError(f"probability cache SHA256 mismatch: {path}")
            if int(record.get("bytes", -1)) != path.stat().st_size:
                raise ValueError(f"probability cache byte-size mismatch: {path}")
            _validate_cached_probability(
                path,
                source,
                checkpoint_sha256=checkpoint_sha256,
                calibration_sha256=calibration["sha256"],
                inference_source_contract_sha256=(
                    inference_source_contract_sha256
                ),
            )
        all_records.extend(records)
        workers.append(_sealed_worker_summary(summary))

    by_cache_id = {str(record["cache_id"]): record for record in all_records}
    expected_cache_ids = [str(row["cache_id"]) for row in rows]
    if len(by_cache_id) != len(all_records):
        raise ValueError("worker manifests contain duplicate cache IDs")
    if set(by_cache_id) != set(expected_cache_ids):
        raise ValueError("worker manifests do not exactly cover the full roster")

    manifest_rows = []
    for row in rows:
        record = by_cache_id[str(row["cache_id"])]
        manifest_rows.append(
            {
                "cache_id": row["cache_id"],
                "sample_id": row["sample_id"],
                "role": row["role"],
                "split": row["split"],
                "probability_path": record["probability_path"],
                "probability_sha256": record["probability_sha256"],
                "inference_source_contract_sha256": (
                    inference_source_contract_sha256
                ),
                "bytes": int(record["bytes"]),
                "contains_ground_truth": False,
            }
        )
    # ``full_cache.py`` and its locked launcher consume this canonical name.
    manifest_path = root / "predictions.jsonl"
    atomic_jsonl_dump(manifest_rows, manifest_path)
    completion = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "samples": len(manifest_rows),
        "role_counts": dict(role_counts),
        "world_size": world_size,
        "canvas_hw": list(CANVAS_HW),
        "probability_dtype": "float16",
        "probability_definition": "sigmoid((mask_logits[channel1]-mask_logits[channel0])/temperature)",
        "padding_policy": "probability is zero outside the letterboxed RGB content",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "roster": str(roster_file),
        "roster_sha256": roster_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration": str(calibration_file),
        "calibration_sha256": calibration["sha256"],
        "inference_source_contract": inference_source_contract,
        "inference_source_contract_sha256": inference_source_contract_sha256,
        "temperature": calibration["temperature"],
        "gate_threshold": calibration["gate_threshold"],
        "bytes": sum(int(row["bytes"]) for row in manifest_rows),
        "rgb_access": True,
        "ground_truth_mask_access": False,
        "ground_truth_depth_access": False,
        "contains_ground_truth": False,
        "strict_roster_coverage": True,
        "workers": workers,
    }
    atomic_json_dump(completion, root / "complete.json")
    return completion


def merge_tomnet_masks(
    config_path: str | Path,
    roster_path: str | Path,
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    *,
    world_size: int = DEFAULT_WORLD_SIZE,
) -> dict[str, Any]:
    """Seal the official 46,679-row full-roster probability manifest."""

    return _merge_tomnet_masks(
        config_path,
        roster_path,
        checkpoint_path,
        calibration_path,
        world_size=world_size,
        expected_role_counts=EXPECTED_ROLE_COUNTS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Eight-way RGB-only TOM-Net probability inference"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("worker", "merge"):
        item = subparsers.add_parser(command)
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--roster", type=Path, required=True)
        item.add_argument("--checkpoint", type=Path, required=True)
        item.add_argument("--calibration", type=Path, required=True)
        item.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
        if command == "worker":
            item.add_argument("--rank", type=int, required=True)
            item.add_argument(
                "--device",
                help="defaults to cuda:<rank>; pass cpu only for tests/smoke checks",
            )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "worker":
        result = infer_worker(
            args.config,
            args.roster,
            args.checkpoint,
            args.calibration,
            rank=args.rank,
            world_size=args.world_size,
            device_name=args.device,
        )
    else:
        result = merge_tomnet_masks(
            args.config,
            args.roster,
            args.checkpoint,
            args.calibration,
            world_size=args.world_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
