"""Record portable Stage1 code identity after reviewed changes, before parity.

This is a source inventory, not a successful execution receipt. GPU acceptance
must bind this file's SHA and be rerun after changes to the inventory.
"""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    roots = ["src/native_fusion", "src/laplacian_train", "src/laplacian_fusion",
             "third_party/moge", "third_party/dinov3", "third_party/transnormal", "third_party/stage1_support"]
    files = [p for root in roots for p in (PROJECT/root).rglob("*.py")]
    files += [PROJECT/"src/transparent_depth"/name for name in
              ("stage1.py","stage1_experts.py","stage1_batch.py","input_guard.py")]
    files += [PROJECT/"scripts"/name for name in
              ("trace_stage1_live.py","compare_stage1_traces.py","prepare_stage1_relocation.py",
               "init_stage1_training.py","setup_stage1.sh")]
    files += [
              PROJECT/"provenance/stage1_assets.json", PROJECT/"provenance/stage1_seed_identities.json",
              PROJECT/"requirements-stage1.txt",PROJECT/"third_party/cleargrasp/data_processing_script.py"]
    files += list((PROJECT/"provenance/stage1_historical").glob("*.json"))
    record={"schema":"portable_stage1_source_inventory_v1","execution_verified":False,
            "sha256":{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}}
    (PROJECT/"provenance/stage1_runtime_sources.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("Source inventory written. This does not certify numerical parity.")


if __name__=="__main__":main()
