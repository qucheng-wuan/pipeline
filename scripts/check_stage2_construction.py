"""Real initialization/runtime preflight, not a pipeline or training smoke.

The unchanged official SPVNAS constructor performs its own sparse dummy forward
to specialize the topology. No RGB, P, GT, mask or producer seal is fabricated.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    import torch
    from transparent_depth.stage2 import build_stage2, source_hashes, file_sha
    from v2_lidf.data import atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/cleargrasp256_stage2_smoke.json'))
    parser.add_argument('--assets-root',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new; keep earlier failure evidence')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    cfg = json.loads(Path(args.config).read_text())
    report = {'schema':'stage2_construction_preflight_v1','config_sha256':file_sha(args.config),
        'source_sha256':source_hashes(), 'pipeline_forward_executed':False,
        'backward_executed':False,'optimizer_update_executed':False,'end_to_end_passed':False,
        'stage1_predictions_generated':False,'official_topology_dummy_forward_attempted':True}
    started = time.monotonic()
    try:
        model = build_stage2(cfg['model'],args.device,initialization=True,
            assets_root=args.assets_root,audit_dir=args.output/'initialization')
        report.update(status='construction_passed',input_hw=list(model.input_hw),runtime=model.runtime_audit,
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model
        torch.cuda.empty_cache()
    except Exception:
        report.update(status='blocked',error=traceback.format_exc())
    report['seconds'] = time.monotonic()-started
    atomic_json(args.output/'report.json',report)
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='construction_passed' else 2


if __name__=='__main__':
    raise SystemExit(main())
