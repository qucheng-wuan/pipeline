"""Observable file-access isolation for RGB-only acceptance and batch export.

This is an engineering tripwire, not an OS sandbox. Python audit events cover
Python and extension-mediated opens that report those events; explicit image
reader wrappers cover OpenCV and OpenEXR. Dependencies and native system library
loading remain permitted. The receipt states that scope rather than claiming a
kernel-level guarantee.
"""
from __future__ import annotations
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from unittest.mock import patch


def _within(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)


class RGBInputGuard:
    def __init__(self, *, project, asset_root, rgb_paths, output_root, cache_root,
                 metadata_paths=(), extra_runtime_roots=(), deny_prefixes=()):
        self.project=Path(project).resolve()
        self.asset_root=Path(asset_root).resolve()
        self.rgb={Path(p).resolve() for p in rgb_paths}
        self.metadata={Path(p).resolve() for p in metadata_paths}
        self.output_root=Path(output_root).resolve()
        self.cache_root=Path(cache_root).resolve()
        self.runtime_roots={Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),
            *[Path(p).resolve() for p in extra_runtime_roots]}
        # Driver/runtime resources; no user dataset directory is allowed here.
        self.system_roots={Path(p) for p in ('/usr','/lib','/lib64','/etc','/proc','/sys','/dev')}
        self.allowed_roots={self.project,self.asset_root,self.output_root,self.cache_root,
                            *self.runtime_roots,*self.system_roots}
        self.denied_roots={Path(p).resolve() for p in deny_prefixes}
        self.active=False
        self.reads=set()
        self.denials=[]
        self.stack=None

    def check(self, raw, *, read=True):
        if not self.active or isinstance(raw,int) or raw is None:
            return
        try:
            path=Path(os.fsdecode(raw)).expanduser().resolve()
        except (TypeError,ValueError):
            return
        if not read:
            return
        name=str(path).lower()
        label=(name.endswith('.exr') or '-segmentation-mask.' in name or
               '-opaque-depth-img.' in name or name.endswith('-mask.png')
               or path.name.lower() in {'depth1.png','depth2.png','depth1-gt.png','depth2-gt.png'})
        if label or _within(path,self.denied_roots):
            self.denials.append(str(path))
            raise PermissionError('RGB-only file-access guard denied: '+str(path))
        # Only explicitly named RGB/metadata files may lie outside the whitelist.
        if path in self.rgb or path in self.metadata:
            self.reads.add(str(path));return
        if not _within(path,self.allowed_roots):
            self.denials.append(str(path))
            raise PermissionError('RGB-only file-access guard denied: '+str(path))
        self.reads.add(str(path))

    def _audit(self,event,args):
        if event=='open' and self.active:
            path,mode,flags=args
            reading=(not isinstance(mode,str) or 'r' in mode or '+' in mode)
            if mode is None and isinstance(flags,int):
                reading=(flags & os.O_ACCMODE)!=os.O_WRONLY
            self.check(path,read=reading)

    def __enter__(self):
        self.cache_root.mkdir(parents=True,exist_ok=True)
        self.stack=ExitStack()
        # Imports occur before enabling the hook; relevant source locations are
        # separately verified and every subsequent prediction read is observed.
        try:
            import cv2
        except ImportError:
            cv2=None
        if cv2 is not None:
            original=cv2.imread
            def imread(path,*args,**kwargs):
                self.check(path);return original(path,*args,**kwargs)
            self.stack.enter_context(patch.object(cv2,'imread',imread))
        try:
            import OpenEXR
        except ImportError:
            OpenEXR=None
        if OpenEXR is not None:
            original_exr=OpenEXR.File
            def exr(path,*args,**kwargs):
                self.check(path);return original_exr(path,*args,**kwargs)
            self.stack.enter_context(patch.object(OpenEXR,'File',exr))
        sys.addaudithook(self._audit)
        self.active=True
        return self

    def __exit__(self,*args):
        self.active=False
        return self.stack.__exit__(*args)

    def receipt(self):
        return {'schema':'rgb_only_open_guard_v1','scope':'Python audit open events + OpenCV/OpenEXR wrappers; not OS sandbox',
            'project':str(self.project),'asset_root':str(self.asset_root),
            'allowed_rgb_count':len(self.rgb),'read_paths':sorted(self.reads),
            'denied_paths':self.denials,'ground_truth_read':False,
            'legacy_cache_read':False,'strict_read_allowlist':True}


def assert_project_modules(project):
    root=Path(project).resolve()
    prefixes=('transparent_depth','native_fusion','laplacian_train','laplacian_fusion',
              'far_r5','moge_transparent_fusion','moge','dinov3','dinov3_features','transnormal')
    result={}
    for name,module in list(sys.modules.items()):
        if not any(name==p or name.startswith(p+'.') for p in prefixes):continue
        filename=getattr(module,'__file__',None)
        if filename is None:continue
        path=Path(filename).resolve()
        if not path.is_relative_to(root):raise RuntimeError('foreign Stage1 module loaded: '+name+' '+str(path))
        result[name]=str(path.relative_to(root))
    return result


def configure_jit_cache(root):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    os.environ['TD_STAGE1_RUNTIME_CACHE']=str(root)
    # All are cache/output directories, never model or dataset input roots.
    for variable,child in [('TRITON_CACHE_DIR','triton'),('TORCH_EXTENSIONS_DIR','torch_extensions'),
                           ('CUDA_CACHE_PATH','cuda'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
        os.environ[variable]=str(root/child)
        (root/child).mkdir(parents=True,exist_ok=True)
    import tempfile
    tempfile.tempdir=str(root/'tmp')
    return root
