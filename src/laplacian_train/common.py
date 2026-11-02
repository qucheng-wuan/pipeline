import hashlib
import json
import os
from pathlib import Path
import torch

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('TD_STAGE1_LAP_ROOT', 'artifacts/stage1_training/laplacian')).resolve()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def save_checkpoint(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,tmp)
    os.replace(tmp,path)


def move(value,device):
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    return value.to(device,non_blocking=True) if isinstance(value,torch.Tensor) else value


def numerics():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def source_hashes():
    return {'src/laplacian_train/'+name:sha(PROJECT/'src/laplacian_train'/name)
            for name in ('model.py','losses.py','objective.py','data.py','prepare.py','common.py','train.py')}


def check_protocol(root=ROOT):
    p=read(root/'protocol.json')
    if p['source_sha256']!=source_hashes():raise ValueError('frozen implementation changed')
    return p
