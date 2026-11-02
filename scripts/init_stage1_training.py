"""Initialize a fresh portable Laplacian reproduction protocol, never an old run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from laplacian_train.common import source_hashes, write
from transparent_depth.stage1 import Stage1Assets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    a=Stage1Assets()
    protocol={'schema':'portable_stage1_laplacian_reproduction_v1','source_sha256':source_hashes(),
        'frozen_v2_sha256':a.entries['v2']['sha256'],'frozen_e08_reference_sha256':a.entries['laplacian_e08']['sha256'],
        'assets_sha256':a.manifest_sha256,'epochs':8,'seed':20261023,'pyramid_weight':0.05,
        'checkpoint_reference_is_unchanged':True,'new_training_is_not_bitwise_trajectory_claim':True}
    path=args.root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise FileExistsError('different existing protocol')
    write(path,protocol)
    write(args.root/'promotion.json',{'promoted':True,'pyramid_weight':0.05,
        'reason':'explicit reproduction of already selected full_pyramid protocol; no new pilot selection'})
    print(path)


if __name__=='__main__':main()
