"""Separated RGB/K inputs and ClearGrasp supervision, compatible with Python 3.8.

Released depth-imgs-rectified and opaque-depth-img EXRs already store optical-Z
in metres. Applying another radial-to-Z conversion or unit scale is incorrect.
The stage-two camera convention uses integer array pixel centres (u, v).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MODEL_HW = (240, 320)
INPUT_SCHEMA = "v2_lidf_rgb_calibration_input_v1"
LABEL_SCHEMA = "v2_lidf_separate_supervision_v1"
LABEL_CACHE_SCHEMA = "v2_lidf_curriculum_rectified_metric_labels_v1"
INPUT_FIELDS = frozenset({
    "schema_version", "id", "item_id", "cache_id", "sample_id", "role", "domain",
    "official_split", "benchmark_group", "rgb_path", "rgb_sha256", "native_hw",
    "model_hw", "intrinsics_native", "intrinsics", "intrinsics_pixel_convention",
    "intrinsics_source", "intrinsics_source_sha256", "intrinsics_source_kind",
    "unit", "coordinate", "dataset", "scene", "camera", "preprocessing", "split_sha256",
})


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(resolve_data_path(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_data_path(path):
    """Expand portable dataset root bindings; never silently retain unset vars."""
    value = os.path.expandvars(str(path))
    if '${' in value:
        raise ValueError('Unbound dataset root in path: ' + str(path))
    return os.path.expanduser(value)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".%d.tmp" % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".%d.tmp" % os.getpid())
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(str(temporary), str(path))


def boundary_to_integer_intrinsics(intrinsics):
    result = np.asarray(intrinsics, dtype=np.float64).copy()
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("intrinsics must be finite 3x3")
    result[0, 2] -= .5
    result[1, 2] -= .5
    return result


def resize_intrinsics(intrinsics, native_hw, model_hw=MODEL_HW):
    """align_corners=False resize: c'=(c+.5)*scale-.5, not c'=c*scale."""
    h, w = native_hw
    oh, ow = model_hw
    if min(h, w, oh, ow) <= 0:
        raise ValueError("positive image dimensions required")
    transform = np.array([[ow / w, 0., (ow / w - 1.) / 2.],
                          [0., oh / h, (oh / h - 1.) / 2.], [0., 0., 1.]])
    return transform @ np.asarray(intrinsics, dtype=np.float64)


def validate_input_row(row):
    extras = set(row) - INPUT_FIELDS
    if extras:
        raise ValueError("input manifest contains unapproved/label fields: %s" % sorted(extras))
    if row.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("incompatible RGB/K input schema")
    if row.get("role") not in {"train", "val", "evaluation"}:
        raise ValueError("unsupported input role")
    identifier = row["cache_id"]
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("unsafe cache identity")
    if row["item_id"] != identifier or row["id"] != identifier:
        raise ValueError("inconsistent cache/item identity")
    if row.get("unit") != "m" or row.get("coordinate") != "optical_z":
        raise ValueError("metric optical-Z required")
    if row.get("intrinsics_pixel_convention") != "integer_pixel_centres":
        raise ValueError("integer camera pixel centres required")
    if len(row["rgb_sha256"]) != 64:
        raise ValueError("RGB SHA256 required")
    if 'preprocessing' in row:
        from .data_protocol import validate_protocol
        protocol = validate_protocol(row['preprocessing'])
        if row['model_hw'] != protocol['model_hw'] or not row.get('split_sha256'):
            raise ValueError('256 geometry/split protocol mismatch')
    expected = resize_intrinsics(row["intrinsics_native"], row["native_hw"], row["model_hw"])
    if not np.allclose(expected, row["intrinsics"], rtol=1e-7, atol=1e-6):
        raise ValueError("resized camera intrinsics mismatch")
    if np.any(np.diag(expected) <= 0) or not np.allclose(expected[2], [0., 0., 1.]):
        raise ValueError("invalid calibration")


def load_input_manifest(path, role=None):
    rows = read_jsonl(path)
    seen = set()
    identities = {}
    for row in rows:
        validate_input_row(row)
        key = (row["role"], row["cache_id"])
        if key in seen:
            raise ValueError("duplicate input identity within role")
        seen.add(key)
        signature = {k: v for k, v in row.items() if k != "role"}
        if row["cache_id"] in identities and identities[row["cache_id"]] != signature:
            raise ValueError("shared cache identity has inconsistent input")
        identities[row["cache_id"]] = signature
    return rows if role is None else [r for r in rows if r["role"] == role]


def load_label_manifest(path, role=None):
    rows = read_jsonl(path)
    if any(row.get("schema_version") != LABEL_SCHEMA for row in rows):
        raise ValueError("incompatible supervision schema")
    if len({(r["role"], r["item_id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate supervision identity within role")
    return rows if role is None else [r for r in rows if r["role"] == role]


def read_exr_channels(path, names=("R",)):
    """Use the common OpenEXR API available in both existing environments."""
    import OpenEXR
    import Imath
    source = OpenEXR.InputFile(resolve_data_path(path))
    try:
        window = source.header()["dataWindow"]
        shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        arrays = [np.frombuffer(source.channel(name, pixel_type), dtype=np.float32)
                  .reshape(shape).copy() for name in names]
        return arrays
    finally:
        source.close()


def load_native_labels(label_row, include_normals=False):
    """Label-aware loader. No rescale or second depth rectification is applied."""
    from PIL import Image
    semantics = label_row.get("depth_semantics")
    if semantics == 'transcg_uint16_optical_z':
        camera = label_row.get('camera')
        expected = {'D435': 1000., 'L515': 4000.}
        divisor = label_row.get('depth_divisor')
        if camera not in expected or divisor != expected[camera]:
            raise ValueError('transCG camera/unit contract mismatch')
        with Image.open(resolve_data_path(label_row['target_depth_path'])) as image:
            raw = np.asarray(image).copy()
        if raw.dtype not in (np.uint16, np.int32) or raw.ndim != 2 or np.any(raw < 0) or np.any(raw > 65535):
            raise ValueError('transCG GT must be uint16 PNG')
        depth = raw.astype(np.float32) / np.float32(divisor)
        mask_values = [0, 1]
    elif semantics == "already_rectified_optical_z_m":
        depth = read_exr_channels(label_row["target_depth_path"])[0]
        mask_values = [0, 255]
    else:
        raise ValueError("unverified target depth semantics")
    if tuple(depth.shape) != tuple(label_row["native_hw"]):
        raise ValueError("native target shape mismatch")
    with Image.open(resolve_data_path(label_row["transparent_mask_path"])) as image:
        if image.mode != "L":
            raise ValueError("transparent mask must be grayscale")
        mask = np.asarray(image).copy()
    if mask.shape != depth.shape or not np.isin(mask, mask_values).all():
        raise ValueError("mask shape or declared encoding mismatch: %s" % mask_values)
    valid = np.isfinite(depth) & (depth > 0)
    result = {"target": np.where(valid, depth, 0.).astype(np.float32),
              "valid": valid, "transparent": mask == mask_values[1],
              "intrinsics": np.asarray(label_row["intrinsics_native"], dtype=np.float64)}
    if include_normals and label_row.get("normal_path"):
        raw = np.stack(read_exr_channels(label_row["normal_path"], ("R", "G", "B")))
        finite = np.isfinite(raw).all(0)
        clean = np.where(np.isfinite(raw), raw, 0.)
        norm = np.linalg.norm(clean, axis=0)
        support = finite & ~(clean == -1).all(0) & (norm >= .5) & (norm <= 1.5)
        normal = clean * np.array([1., -1., -1.], dtype=np.float32)[:, None, None]
        result.update(normal=(normal / np.maximum(norm, 1e-6)) * support[None],
                      normal_valid=support)
    return result


def nearest_exact_resize(array, model_hw=MODEL_HW):
    """Pixel-centre nearest resize matching torch mode='nearest-exact'."""
    h, w = array.shape[-2:]
    oh, ow = model_hw
    yi = np.minimum(((np.arange(oh) + .5) * h / oh).astype(np.int64), h - 1)
    xi = np.minimum(((np.arange(ow) + .5) * w / ow).astype(np.int64), w - 1)
    return array[..., yi[:, None], xi[None, :]].copy()


def source_stat(path):
    source = Path(resolve_data_path(path))
    stat = source.stat()
    if str(path).startswith('${'):
        return {'path': str(path), 'bytes': stat.st_size, 'sha256': sha256_file(source)}
    return {"path": str(source), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def label_cache_identity(row):
    identity = {"schema_version": LABEL_CACHE_SCHEMA, "item_id": row["item_id"],
            "rgb_sha256": row["rgb_sha256"], "model_hw": row["model_hw"],
            "role": row["role"],
            "training_background_gt_used_as_model_input": row["role"] == "train",
            "training_gt_mask_used_for_curriculum": row["role"] == "train",
            "depth_semantics": row["depth_semantics"],
            "sources": {"depth": source_stat(row["target_depth_path"]),
                        "transparent_mask": source_stat(row["transparent_mask_path"])}}
    if 'preprocessing' in row:
        identity.update(preprocessing=row['preprocessing'], split_sha256=row['split_sha256'],
                        intrinsics_native=row['intrinsics_native'], camera=row['camera'],
                        depth_divisor=row.get('depth_divisor', 1.))
    return identity


def prepare_label_cache(row, label_root, overwrite=False):
    if row["role"] not in {"train", "val"}:
        raise ValueError("only train/val labels may enter optimization cache")
    path = Path(label_root) / (row["cache_id"] + ".npz")
    identity = label_cache_identity(row)
    if path.is_file():
        with np.load(str(path), allow_pickle=False) as payload:
            previous = json.loads(str(payload["metadata_json"].item()))
        if previous.get("identity", {}).get("schema_version") != LABEL_CACHE_SCHEMA:
            raise ValueError("refusing to overwrite another label-cache schema; use a new label_root")
    if path.is_file() and not overwrite:
        with np.load(str(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata["identity"] != identity:
                raise ValueError("cached supervision source changed: %s" % path)
            if payload["depth_gt"].shape != tuple(row["model_hw"]):
                raise ValueError("cached target resolution mismatch")
            if 'preprocessing' in row:
                if ('coverage' not in payload or payload['coverage'].dtype != np.float32
                        or payload['coverage'].shape != tuple(row['model_hw'])):
                    raise ValueError('Weighted-area cache requires float32 coverage')
            if row["role"] == "train":
                if ("transparent_mask" not in payload
                        or payload["transparent_mask"].shape != tuple(row["model_hw"])
                        or payload["transparent_mask"].dtype != np.bool_):
                    raise ValueError("training curriculum cache needs an aligned boolean mask")
            elif "transparent_mask" in payload:
                raise ValueError("validation cache must not expose a curriculum mask")
        return {"item_id": row["item_id"], "status": "existing", "path": str(path)}
    native = load_native_labels(row)
    extra = {}
    if 'preprocessing' in row:
        from .data_protocol import resize_supervision
        resized = resize_supervision(native, row['preprocessing'])
        depth, valid = resized['target'], resized['valid']
        extra['coverage'] = resized['coverage']
    else:
        depth = nearest_exact_resize(native["target"], row["model_hw"])
        valid = nearest_exact_resize(native["valid"], row["model_hw"])
    training = row["role"] == "train"
    metadata = {"identity": identity,
                "source_sha256": {k: sha256_file(v["path"]) for k, v in identity["sources"].items()},
                "resize": row.get('preprocessing', {}).get('gt_interpolation', 'nearest_exact_pixel_centres'),
                "ground_truth_access": True,
                "gt_mask_used_as_input": training,
                "training_background_gt_used_as_model_input": training,
                "training_gt_mask_used_for_curriculum": training,
                "transparent_gt_depth_used_as_model_input": False,
                "validation_or_inference_input": "frozen_rgb_k_predictor_only"}
    if training:
        extra['transparent_mask'] = nearest_exact_resize(native['transparent'], row['model_hw']).astype(bool)
    atomic_npz(path, depth_gt=depth.astype(np.float32), gt_valid=valid.astype(bool),
               metadata_json=np.array(json.dumps(metadata, sort_keys=True)), **extra)
    return {"item_id": row["item_id"], "status": "created", "path": str(path)}


class V2LIDFDataset(Dataset):
    """Read authenticated RGB-derived D1 and separately authorized supervision.

    ``curriculum=True`` exposes a training-only mask to the training loop. The
    dataset itself never mixes inputs; alpha follows committed optimizer steps
    in the main process, independent of worker prefetch and checkpoint resume.
    """
    def __init__(self, input_manifest, label_manifest, stage1_root, label_root,
                 role="train", include_target=True, curriculum=False):
        if curriculum and (role != "train" or not include_target):
            raise ValueError("curriculum masks are permitted only for supervised training")
        self.curriculum = bool(curriculum)
        self.rows = load_input_manifest(input_manifest, role)
        self.stage1_root = Path(stage1_root)
        self.label_root = Path(label_root) if label_root is not None else None
        self.include_target = include_target
        self.sealed_inputs = None
        marker_path = self.stage1_root / "complete.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            receipts_path = self.stage1_root / "inputs.sealed.jsonl"
            if (marker.get("status") != "complete"
                    or marker.get("input_manifest_sha256") != sha256_file(input_manifest)
                    or marker.get("sealed_inputs_manifest_sha256") != sha256_file(receipts_path)
                    or marker.get("all_arrays_sha256_verified") is not True):
                raise ValueError("stage-one seal differs from current inputs or receipts")
            receipts = read_jsonl(receipts_path)
            self.sealed_inputs = {r["cache_id"]: r for r in receipts}
            if len(self.sealed_inputs) != len(receipts):
                raise ValueError("stage-one seal contains duplicate cache identities")
            for row in self.rows:
                receipt = self.sealed_inputs.get(row["cache_id"])
                if receipt is None or receipt["rgb_sha256"] != row["rgb_sha256"]:
                    raise ValueError("stage-one sealed receipt does not match this dataset")
        self.label_rows = {}
        if include_target:
            if role not in {"train", "val"}:
                raise ValueError("evaluation labels are loaded separately after prediction")
            self.label_rows = {r["item_id"]: r for r in load_label_manifest(label_manifest, role)}
            if set(self.label_rows) != {r["item_id"] for r in self.rows}:
                raise ValueError("input and supervision identities differ")
        self.mean = torch.tensor([.485, .456, .406])[:, None, None]
        self.std = torch.tensor([.229, .224, .225])[:, None, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.stage1_root / "inputs" / (row["cache_id"] + ".npz")
        if (self.sealed_inputs is not None
                and sha256_file(path) != self.sealed_inputs[row["cache_id"]]["input_sha256"]):
            raise ValueError("stage-one input changed after sealing: %s" % row["cache_id"])
        with np.load(str(path), allow_pickle=False) as payload:
            rgb = np.array(payload["rgb"], copy=True)
            depth = np.array(payload["depth_input"], dtype=np.float32, copy=True)
            intrinsics = np.array(payload["intrinsics"], dtype=np.float32, copy=True)
            metadata = json.loads(str(payload["metadata_json"].item()))
        if rgb.dtype != np.uint8 or rgb.shape != (*tuple(row["model_hw"]), 3):
            raise ValueError("stage-one RGB must be uint8 model-resolution HWC")
        if depth.shape != tuple(row["model_hw"]) or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError("stage-one depth must be finite positive metric HW")
        if not np.allclose(intrinsics, row["intrinsics"], atol=1e-5, rtol=1e-6):
            raise ValueError("stage-one calibration differs from manifest")
        if metadata.get("rgb_sha256") != row["rgb_sha256"]:
            raise ValueError("stage-one RGB identity differs from manifest")
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.
        result = {"rgb": (rgb_tensor - self.mean) / self.std,
                  "depth_input": torch.from_numpy(depth)[None],
                  "intrinsics": torch.from_numpy(intrinsics),
                  "item_id": row["item_id"], "index": index}
        if self.include_target:
            label_row = self.label_rows[row["item_id"]]
            if label_row["rgb_sha256"] != row["rgb_sha256"]:
                raise ValueError("supervision RGB identity differs from input")
            label_path = self.label_root / (row["cache_id"] + ".npz")
            with np.load(str(label_path), allow_pickle=False) as payload:
                depth_gt = np.array(payload["depth_gt"], dtype=np.float32, copy=True)
                valid = np.array(payload["gt_valid"], dtype=bool, copy=True)
                meta = json.loads(str(payload["metadata_json"].item()))
                if 'preprocessing' in row:
                    if (meta['identity'].get('preprocessing') != row['preprocessing']
                            or meta['identity'].get('split_sha256') != row['split_sha256']):
                        raise ValueError('Cached supervision preprocessing/split differs')
                    coverage = np.array(payload['coverage'], dtype=np.float32, copy=True)
                    if coverage.shape != depth.shape or not np.isfinite(coverage).all():
                        raise ValueError('Invalid GT coverage')
                    result['coverage'] = torch.from_numpy(coverage)[None]
                if self.curriculum:
                    if (meta["identity"].get("schema_version") != LABEL_CACHE_SCHEMA
                            or meta["identity"].get("role") != "train"
                            or meta.get("training_gt_mask_used_for_curriculum") is not True
                            or meta.get("transparent_gt_depth_used_as_model_input") is not False):
                        raise ValueError("explicit training curriculum label cache required")
                    transparent = np.array(payload["transparent_mask"], copy=True)
            if meta["identity"]["rgb_sha256"] != row["rgb_sha256"] or meta["identity"]["item_id"] != row["item_id"]:
                raise ValueError("cached supervision identity mismatch")
            if depth_gt.shape != depth.shape or valid.shape != depth.shape:
                raise ValueError("target resolution mismatch")
            if not np.isfinite(depth_gt).all() or np.any(depth_gt[valid] <= 0):
                raise ValueError("invalid target depth")
            result.update(depth_gt=torch.from_numpy(depth_gt)[None],
                          gt_valid=torch.from_numpy(valid)[None])
            if self.curriculum:
                if transparent.shape != depth.shape or transparent.dtype != np.bool_:
                    raise ValueError("curriculum mask must be boolean and aligned with target")
                result["transparent_mask"] = torch.from_numpy(transparent)[None]
        return result
