"""Generate portable frozen RGB-expert inputs for V2 training, without GT reads.

The input JSONL supplies RGB identities/splits, native geometry and label paths;
label paths are carried as metadata and are only opened by train/evaluate.
Workers shard a roster. A separate seal validates every generated tensor before
the original V2 trainer can consume it.
"""
from __future__ import annotations
import argparse
from collections import Counter
import io
import json
import os
from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
from .data import SCHEMA, sha256_file, read_jsonl, resolve_asset, camera_from_rgb
from .cache import producer_contract


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_bytes(value)
    os.replace(temp, path)


def write_json(path, value):
    atomic(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())


def tensor_asset(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    atomic(path, buffer.getvalue())
    return {"path":str(Path(path).resolve()), "sha256":sha256_file(path), "bytes":len(buffer.getvalue())}


def normalize_rows(manifest):
    rows = read_jsonl(manifest)
    seen = set()
    for row in rows:
        if row["cache_id"] in seen or not re.fullmatch(r"[A-Za-z0-9_-]+",row["cache_id"]):
            raise ValueError("duplicate or unsafe cache ID")
        seen.add(row["cache_id"])
        rgb = resolve_asset(row["rgb_path"])
        digest = sha256_file(rgb)
        if row.get("rgb_sha256",digest) != digest:
            raise ValueError("RGB identity changed")
        with Image.open(rgb) as image:
            hw = [image.height,image.width]
        if row.get("native_hw",hw) != hw:
            raise ValueError("native RGB shape differs")
        row.update(schema_version=SCHEMA,rgb_sha256=digest,native_hw=hw,rgb_path=str(rgb.resolve()))
        if "camera" not in row:
            row["camera"] = camera_from_rgb(rgb,tuple(hw))
        if row["camera"].get("pixel_center_offset") != .5:
            raise ValueError("V2 supervision uses boundary-coordinate K (pixel centre u+0.5)")
        if row["split"] not in ("train","val","test") or row["domain"] not in ("real","synthetic"):
            raise ValueError("unknown split/domain")
    return rows


def worker(manifest, output, *, rank=0, world=1, device="cuda:0"):
    from transparent_depth.stage1 import Stage1Predictor, forbid_label_reads
    if not 0 <= rank < world:
        raise ValueError("invalid shard")
    rows = normalize_rows(manifest)
    predictor = Stage1Predictor(device=device)
    for row in rows[rank::world]:
        cid = row["cache_id"]
        receipt_path = output / "prepare_receipts" / (cid+".json")
        identity = {k:row[k] for k in ("sample_id","cache_id","rgb_sha256","native_hw","split","domain")}
        if receipt_path.exists():
            existing = json.loads(receipt_path.read_text())
            if (existing["input_identity"] != identity or existing["assets_sha256"] != predictor.assets.manifest_sha256
                    or existing["sources_sha256"] != predictor.source_sha256
                    or any(existing["row"].get(key)!=value for key,value in row.items())):
                raise ValueError("cannot reuse different prepared input")
            for asset in existing["files"]:
                if sha256_file(asset["path"]) != asset["sha256"]:
                    raise ValueError("prepared asset changed")
            continue
        with forbid_label_reads():
            _, trace = predictor.components(resolve_asset(row["rgb_path"]),return_trace=True)
        tags = {"sample_id":row["sample_id"],"rgb_sha256":row["rgb_sha256"],"contains_ground_truth":False,"ground_truth_access":False}
        compact = tensor_asset(output / "expert_inputs" / (cid+".pt"),
            {**trace["compact"],**tags,"schema_version":"far_r5_compact_inputs_v1"})
        residual = tensor_asset(output / "r5_residuals" / (cid+".pt"),
            {k:trace[k] for k in ("bounded_delta","gate","context_descriptor","context_r5_native")} | tags)
        residual.update(cache_id=cid)
        if row["split"] == "train":
            native = tensor_asset(output / "native_moge" / (cid+".pt"),{**tags,"base_depth_native":trace["moge_native"]})
            native_r5 = None
        else:
            native = save_npy(output / "native_moge" / (cid+".npy"),trace["moge_native"])
            native_r5 = save_npy(output / "native_r5" / (cid+".npy"),trace["r5_native"])
        built = {**row,"compact":compact,"native_moge":native}
        if native_r5 is not None:
            built["native_r5"] = native_r5
        write_json(receipt_path,{"input_identity":identity,"row":built,"residual":residual,
            "assets_sha256":predictor.assets.manifest_sha256,"sources_sha256":predictor.source_sha256,
            "ground_truth_read":False,"files":[compact,residual,native]+([native_r5] if native_r5 else [])})
        print(json.dumps({"phase":"stage1_native_prepare","rank":rank,"sample_id":row["sample_id"]}),flush=True)


def save_npy(path,value):
    buffer=io.BytesIO()
    np.save(buffer,value.detach().float().cpu().numpy(),allow_pickle=False)
    atomic(path,buffer.getvalue())
    return {"path":str(path.resolve()),"sha256":sha256_file(path),"bytes":len(buffer.getvalue())}


def seal(manifest, output):
    from transparent_depth.stage1 import Stage1Assets
    assets=Stage1Assets()
    expected_contract=(assets.manifest_sha256,assets.verify_sources())
    rows=normalize_rows(manifest)
    records=[];residuals=[]; contracts=set()
    for row in rows:
        r=json.loads((output/"prepare_receipts"/(row["cache_id"]+".json")).read_text())
        identity={k:row[k] for k in ("sample_id","cache_id","rgb_sha256","native_hw","split","domain")}
        if (r["input_identity"] != identity or r["ground_truth_read"] is not False
                or any(r["row"].get(key)!=value for key,value in row.items())):
            raise ValueError("prepared image identity/input contract changed")
        contracts.add((r["assets_sha256"],r["sources_sha256"]))
        for asset in r["files"]:
            if Path(asset["path"]).stat().st_size != asset["bytes"] or sha256_file(asset["path"]) != asset["sha256"]:
                raise ValueError("prepared asset changed")
        records.append(r["row"]);residuals.append(r["residual"])
    if contracts!={expected_contract}:
        raise ValueError("expert producer changed across workers")
    path=output/"manifests/dataset.jsonl"
    atomic(path,"".join(json.dumps(r,sort_keys=True)+"\n" for r in records).encode())
    write_json(path.with_name("dataset.complete.json"),{"status":"complete","manifest_sha256":sha256_file(path),
        "samples":len(records),"split_counts":dict(Counter(r["split"] for r in records)),"ground_truth_pixels_opened":False})
    rp=output/"r5_residuals/manifest.jsonl"
    atomic(rp,"".join(json.dumps(r,sort_keys=True)+"\n" for r in residuals).encode())
    write_json(rp.with_name("complete.json"),{"status":"complete","samples":len(records),
        "manifest_sha256":sha256_file(rp),"contract":producer_contract(path),"ground_truth_read":False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("worker","seal"))
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rank",type=int,default=0);parser.add_argument("--world",type=int,default=1)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.action=="worker":worker(args.manifest,args.output,rank=args.rank,world=args.world,device=args.device)
    else:seal(args.manifest,args.output)


if __name__=="__main__":main()
