"""Portable, authenticated RGB -> frozen V2 -> learned Laplacian E08 inference.

Only RGB pixels are prediction inputs. No historical checkout, dataset manifest,
cached expert prediction, camera calibration or target is opened by this module.
Weights retain their original bytes; a separate relative asset registry supplies
their locations. Stage2 uses a separate environment and accepts this optical-Z
depth in metres together with RGB and known camera calibration.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT / "provenance/stage1_assets.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Stage1Assets:
    def __init__(self, manifest=None, asset_root=None):
        self.manifest = Path(manifest or os.environ.get("TD_STAGE1_ASSETS", DEFAULT_MANIFEST)).resolve()
        self.root = Path(asset_root or os.environ.get("TD_ASSET_ROOT", PROJECT / "artifacts")).resolve()
        self.data = json.loads(self.manifest.read_text())
        if self.data.get("schema") != "portable_stage1_assets_v1":
            raise ValueError("unsupported Stage1 asset registry")
        self.entries = {a["id"]: a for a in self.data["assets"]}
        if len(self.entries) != len(self.data["assets"]):
            raise ValueError("duplicate Stage1 asset ID")
        self.manifest_sha256 = sha256(self.manifest)
        self.verified = {}

    def path(self, key):
        relative = Path(self.entries[key]["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("asset paths must be relative to the explicit asset root")
        return self.root / relative

    def verify(self, key):
        p, entry = self.path(key), self.entries[key]
        stat = p.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        # Network filesystems can expose a stale/coarse stat after a same-size
        # rewrite. Assets are checked during construction, never per RGB image;
        # hash their bytes on every explicit verification instead of trusting stat.
        if stat.st_size != entry["bytes"] or sha256(p) != entry["sha256"]:
            raise ValueError("Stage1 asset identity mismatch: " + key)
        after = p.stat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != identity:
            raise ValueError("Stage1 asset changed during verification: " + key)
        self.verified[key] = identity
        return p

    def verify_sources(self):
        path = PROJECT / "provenance/stage1_runtime_sources.json"
        record = json.loads(path.read_text())
        for relative, expected in record["sha256"].items():
            candidate = PROJECT / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts or sha256(candidate) != expected:
                raise ValueError("Stage1 implementation changed: " + relative)
        return sha256(path)

    def json_asset(self, key):
        return json.loads(self.verify(key).read_text())


def configure_imports(assets):
    expected = assets.data["runtime"]
    if f"{sys.version_info.major}.{sys.version_info.minor}" != expected["python"]:
        raise RuntimeError("Stage1 requires its Python 3.11 environment")
    for name in ("torch", "torchvision", "diffusers", "transformers", "huggingface_hub", "tokenizers"):
        actual = str(torch.__version__) if name == "torch" else importlib.metadata.version(name)
        if actual != expected[name]:
            raise RuntimeError(f"Stage1 requires {name}=={expected[name]}, found {actual}")
    roots = {"moge": PROJECT / "third_party/moge", "dinov3": PROJECT / "third_party/dinov3",
             "transnormal": PROJECT / "third_party/transnormal",
             "moge_transparent_fusion": PROJECT / "third_party/stage1_support",
             "far_r5": PROJECT / "third_party/stage1_support"}
    for name, root in roots.items():
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root):
            raise RuntimeError("foreign Stage1 module already imported: " + name)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def prepare_flex_gemm_runtime(assets, runtime_cache=None):
    """Seed an isolated writable kernel-configuration cache before FlexGEMM import.

    This preserves the original measured kernel choices. The frozen asset is
    never used as a writable cache, and HOME is never consulted for its seed.
    One persistent predictor owns this process's FlexGEMM runtime.
    """
    if any(name=='flex_gemm' or name.startswith('flex_gemm.') for name in sys.modules):
        raise RuntimeError('FlexGEMM was imported before the authenticated Stage1 runtime; use a fresh process and one persistent predictor')
    from .input_guard import configure_jit_cache
    root=Path(runtime_cache or os.environ.get('TD_STAGE1_RUNTIME_CACHE',
        PROJECT/'outputs/stage1_runtime'/f'pid-{os.getpid()}')).resolve()
    configure_jit_cache(root)
    seed=assets.verify('flex_gemm_autotune_cache')
    raw=seed.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=assets.entries['flex_gemm_autotune_cache']['sha256']:
        raise ValueError('FlexGEMM seed changed during initialization')
    folder=Path(tempfile.mkdtemp(prefix='flex-gemm-',dir=root))
    target=folder/'autotune_cache.json'
    with target.open('xb') as stream:stream.write(raw)
    if sha256(target)!=digest:raise ValueError('FlexGEMM runtime seed copy differs')
    os.environ['FLEX_GEMM_AUTOTUNE_CACHE_PATH']=str(target)
    os.environ['FLEX_GEMM_USE_AUTOTUNE_CACHE']='1'
    os.environ['FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE']='1'
    os.environ['FLEX_GEMM_AUTOTUNE_MODE']='adaptive'
    return {'schema':'stage1_flex_gemm_seed_v1','asset_id':'flex_gemm_autotune_cache',
        'seed_sha256':digest,'runtime_cache_initial_sha256':digest,'runtime_cache_path':str(target),
        'runtime_root':str(root),'home_cache_read':False,'frozen_asset_is_writable_cache':False,
        'autotune_mode':'adaptive','autosave_to_private_copy':True}


@contextlib.contextmanager
def component_precision(matmul, *, cudnn_tf32):
    previous, cudnn = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision(matmul)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
        torch.backends.cudnn.allow_tf32 = cudnn


@contextlib.contextmanager
def forbid_label_reads():
    """An input guard; metadata and weight files remain permitted."""
    import builtins
    from unittest.mock import patch
    original_open, original_path_open = builtins.open, Path.open
    def check(path):
        if isinstance(path, (str, bytes, Path)):
            name = str(path).lower()
            if (name.endswith(".exr") or "-segmentation-mask." in name or "-opaque-depth-img." in name or name.endswith("-mask.png")
                    or Path(str(path)).name.lower() in {'depth1.png','depth2.png','depth1-gt.png','depth2-gt.png'}):
                raise AssertionError("RGB inference attempted to read a label: " + str(path))
    def opened(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)
    def path_opened(path, *args, **kwargs):
        check(path)
        return original_path_open(path, *args, **kwargs)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(builtins, "open", opened))
        stack.enter_context(patch.object(Path, "open", path_opened))
        try:
            import OpenEXR
        except ImportError:
            OpenEXR = None
        if OpenEXR is not None:
            original_exr = OpenEXR.File
            def exr(path, *args, **kwargs):
                check(path)
                return original_exr(path, *args, **kwargs)
            stack.enter_context(patch.object(OpenEXR, "File", exr))
        yield


def load_r5(assets, device):
    from moge_transparent_fusion.gift_normal_r5_export import (
        GiftNormalR5DepthExport, INFERENCE_CONTRACT, _anchor_state,
        _normalized_settings, _settings_from_plan, _verify_anchor_unchanged)
    from moge_transparent_fusion.gift_normal_r5_model import GiftNormalR5SpatialAdapter, r5_trainable_parameter_count
    plan_path = PROJECT / "provenance/stage1_historical/transnormal_teacher_r5_v1.json"
    plan_sha = assets.data["historical_blobs"][plan_path.name]
    if sha256(plan_path) != plan_sha:
        raise ValueError("original R5 plan bytes changed")
    plan = json.loads(plan_path.read_text())
    checkpoint = torch.load(assets.verify("r5"), map_location="cpu", weights_only=True)
    expected = {"schema_version": "moge_transparent_gift_normal_r5_checkpoint_v1", "plan_sha256": plan_sha,
                "teacher_fingerprint": assets.data["teacher_fingerprint"], "inference_contract": INFERENCE_CONTRACT,
                "normal_head": False}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError("R5 checkpoint identity mismatch: " + key)
    settings = _settings_from_plan(plan)
    if _normalized_settings(checkpoint["settings"]) != asdict(settings):
        raise ValueError("R5 settings changed")
    for name in ("o2", "o4"):
        if checkpoint[name + "_checkpoint_sha256"] != assets.entries[name]["sha256"]:
            raise ValueError("R5 frozen anchor changed")
    model = GiftNormalR5SpatialAdapter(assets.verify("o2"), assets.verify("o4"), settings,
        o2_checkpoint_sha256=assets.entries["o2"]["sha256"], o4_checkpoint_sha256=assets.entries["o4"]["sha256"])
    frozen = {name: _anchor_state(model, name) for name in ("o2_anchor", "o4_anchor")}
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for name, state in frozen.items():
        _verify_anchor_unchanged(model, name, state)
    if r5_trainable_parameter_count(model) != checkpoint["parameter_count"]:
        raise ValueError("R5 parameter count changed")
    model.to(device).eval().requires_grad_(False)
    return GiftNormalR5DepthExport(model, teacher_fingerprint=assets.data["teacher_fingerprint"],
                                  checkpoint_sha256=assets.entries["r5"]["sha256"]).eval()


def load_teacher(assets, device):
    from moge_transparent_fusion.transnormal_teacher_r5 import TeacherAssetContract, COORDINATE_VERSION
    from moge_transparent_fusion.io_utils import canonical_json_sha256
    from transnormal import TransNormalPipeline, create_dino_encoder
    for name in assets.entries:
        if name.startswith("transnormal_") or name.startswith("dinov3"):
            assets.verify(name)
    t = assets.data["teacher"]
    fingerprint = canonical_json_sha256({"schema_version":"moge_transparent_transnormal_teacher_cache_v1",
        "revision":"796f97ad3d82ea2e455bed88afe3c33aca8e115f", "source_commit":assets.data["source_revisions"]["transnormal"],
        "unet_sha256":assets.entries["transnormal_unet"]["sha256"], "vae_sha256":assets.entries["transnormal_vae"]["sha256"],
        "projector_sha256":assets.entries["transnormal_projector"]["sha256"],
        "dino_revision":"c807c9eeea853df70aec4069e6f56b28ddc82acc", "dino_sha256":assets.entries["dinov3"]["sha256"],
        "processing_resolution":t["processing_resolution"], "timestep":t["timestep"], "dtype":t["dtype"],
        "canvas_hw":t["canvas_hw"], "coordinate_version":COORDINATE_VERSION})
    if fingerprint != assets.data["teacher_fingerprint"]:
        raise ValueError("teacher protocol fingerprint changed")
    contract = TeacherAssetContract(model_root=assets.path("transnormal_projector").parent,
        source_root=PROJECT / "third_party/transnormal", dino_root=assets.path("dinov3").parent,
        projector=assets.path("transnormal_projector"), revision="796f97ad3d82ea2e455bed88afe3c33aca8e115f",
        source_commit=assets.data["source_revisions"]["transnormal"], processing_resolution=t["processing_resolution"],
        timestep=t["timestep"], dtype=t["dtype"], canvas_hw=tuple(t["canvas_hw"]), fingerprint=fingerprint)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    encoder = create_dino_encoder(model_name="dinov3_vith16plus", weights_path=str(contract.dino_root),
        projector_path=str(contract.projector), device=device, dtype=torch.bfloat16, freeze_encoder=True)
    pipeline = TransNormalPipeline.from_pretrained(str(contract.model_root), dino_encoder=encoder,
        text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    if not getattr(pipeline, "_use_dino_for_cross_attention", False) or pipeline.dino_encoder is None:
        raise RuntimeError("TransNormal must use the pretrained DINO cross-attention")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, contract


class Stage1Predictor:
    def __init__(self, asset_manifest=None, asset_root=None, *, device="cuda:0", runtime_cache=None):
        self.assets = Stage1Assets(asset_manifest, asset_root)
        self.source_sha256 = self.assets.verify_sources()
        self.runtime_configuration=prepare_flex_gemm_runtime(self.assets,runtime_cache)
        configure_imports(self.assets)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the frozen full RGB expert protocol requires CUDA")
        from laplacian_train.common import numerics
        from native_fusion.model import NativeGeometryFusion, Settings
        from laplacian_train.model import LearnableLaplacianFusion
        numerics()
        torch.cuda.set_device(self.device)
        state = torch.load(self.assets.verify("v2"), map_location="cpu", weights_only=True)
        if state.get("schema_version") != "native_geometry_fusion_v2":
            raise ValueError("wrong V2 checkpoint schema")
        self.model = NativeGeometryFusion(Settings(**state["model_settings"])).to(self.device)
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.model.eval().requires_grad_(False)
        lap = torch.load(self.assets.verify("laplacian_e08"), map_location="cpu", weights_only=False)
        self.adapter = LearnableLaplacianFusion(lap.get("model_config", lap.get("config", {}).get("model_settings"))).to(self.device)
        self.adapter.load_state_dict(lap["model"], strict=True)
        self.adapter.eval().requires_grad_(False)
        for module in (self.model, self.adapter):
            if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
                raise ValueError("nonfinite frozen model weights")
        seed = self.assets.data["seed_identities"]
        seed_path = PROJECT / seed["path"]
        if sha256(seed_path) != seed["sha256"]:
            raise ValueError("teacher RNG identity map changed")
        self.seed_identities = json.loads(seed_path.read_text())["identities"]
        self.experts = None

    def load_experts(self):
        if self.experts is not None:
            return self.experts
        from far_r5.anchors import FrozenFARAnchor
        from moge.model.v3 import MoGeModel
        from dinov3_features import DinoV3Features, load_dinov3_h16plus
        from moge_transparent_fusion.tomnet_infer import load_inference_model
        a, device = self.assets, self.device
        r5 = load_r5(a, device)
        calibration = a.json_asset("tom_calibration")
        if calibration.get("checkpoint_sha256") != a.entries["tom"]["sha256"]:
            raise ValueError("TOM calibration belongs to another model")
        tomnet, metadata = load_inference_model(a.verify("tom"), device)
        teacher, contract = load_teacher(a, device)
        anchor = FrozenFARAnchor.from_checkpoint(a.verify("far"), expected_sha256=a.entries["far"]["sha256"], device=device)
        moge = MoGeModel.from_pretrained(a.verify("moge")).to(device).eval().requires_grad_(False)
        dino = DinoV3Features(load_dinov3_h16plus(PROJECT / "third_party/dinov3", a.verify("dinov3"), a.entries["dinov3"]["sha256"])).to(device).eval().requires_grad_(False)
        self.experts = SimpleNamespace(device=device, r5=r5, anchor=anchor, moge=moge, dino=dino,
            tomnet=tomnet, tom_metadata=metadata, calibration=calibration, teacher=teacher,
            teacher_contract=contract, seed_identities=self.seed_identities)
        return self.experts

    @torch.inference_mode()
    def components(self, rgb_path, *, return_trace=False):
        from .stage1_experts import live_components
        with forbid_label_reads():
            return live_components(rgb_path, self.load_experts(), return_trace=return_trace)

    @torch.inference_mode()
    def prepared(self, rgb_path):
        from native_fusion.runtime import predict_native
        from laplacian_train.model import fixed_depth_and_bands, build_gate_context, mean_coefficients
        from .stage1_experts import _infer_moge
        source = self.components(rgb_path)
        v2 = predict_native(self.model, {k:v[None] for k,v in source.items()}, tile_size=384, halo=16, amp=True)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        # Preserve the audited live E08 path, including its second native MoGe call.
        with component_precision("high", cudnn_tf32=True):
            moge = torch.from_numpy(_infer_moge(self.experts.moge, rgb, self.device).copy())
        moge = moge[None,None].to(self.device)
        fixed, bands = fixed_depth_and_bands(v2, moge)
        context = build_gate_context(*(source[k][None].to(self.device) for k in
            ("rgb","probability","teacher_normal","teacher_confidence")), v2, moge, bands)
        return {"fixed_depth":fixed,"bands":bands,"context":context,"mean_coeff":mean_coefficients(bands),
                "v2_depth":v2,"moge_depth":moge,"probability":source["probability"][None].to(self.device)}

    @torch.inference_mode()
    def predict(self, rgb_path):
        from laplacian_train.common import numerics
        from laplacian_train.model import apply_gates_full
        numerics()
        with forbid_label_reads():
            inputs = self.prepared(rgb_path)
            gates = self.adapter(inputs["context"])["gate_delta"]
            output = apply_gates_full(inputs, gates)
        depth = output["depth"][0,0].cpu()
        if not torch.isfinite(depth).all() or not (depth > 0).all():
            raise FloatingPointError("invalid Stage1 metric optical-Z depth")
        return depth.numpy(), {"schema":"portable_stage1_prediction_v1","mode":"live_rgb_experts",
            "force_live":True,"rgb_sha256":sha256(rgb_path),"shape_hw":list(depth.shape),
            "unit":"m","coordinate":"optical_z","external_inputs":["RGB"],
            "camera_intrinsics_input":False,"ground_truth_access":False,"expert_cache_used":False,
            "assets_sha256":self.assets.manifest_sha256,"source_manifest_sha256":self.source_sha256,
            "weight_sha256":{k:a["sha256"] for k,a in self.assets.entries.items()},
            "runtime_configuration":self.runtime_configuration,
            "mean_alpha":(.5+gates.mean((0,2,3))).tolist(),
            "full_log_correction_mean":float(output["log_correction"].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-cache",type=Path,help="Writable private JIT/cache root; default: checkout outputs/stage1_runtime/pid-PID")
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("--output must end with .npy")
    predictor = Stage1Predictor(args.assets, args.asset_root, device=args.device,runtime_cache=args.runtime_cache)
    depth, metadata = predictor.predict(args.rgb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, depth, allow_pickle=False)
    os.replace(temporary, args.output)
    metadata["output_sha256"] = sha256(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status":"complete","output":str(args.output),"shape_hw":metadata["shape_hw"]}))


if __name__ == "__main__":
    main()
