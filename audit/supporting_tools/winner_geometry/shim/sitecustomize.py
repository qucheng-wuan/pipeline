"""Task-only, read-only OpenEXR 3.2.4 File adapter for frozen label readers.

Enabled only with TD_OPENEXR_FILE_COMPAT=1. No numerical operation or package
installation is changed. Only the API used by the frozen depth/normal readers
is implemented; unknown methods and keyword arguments fail normally.
"""
import os

if os.environ.get('TD_OPENEXR_FILE_COMPAT') == '1':
    import atexit
    import hashlib
    import importlib.metadata
    import json
    from pathlib import Path
    import sys
    import time
    from types import SimpleNamespace

    import Imath
    import numpy as np
    import OpenEXR

    version = importlib.metadata.version('OpenEXR')
    if version != '3.2.4' or hasattr(OpenEXR, 'File'):
        raise RuntimeError('This compatibility adapter is only for unmodified OpenEXR 3.2.4 without File')
    shim_path = Path(__file__).resolve()
    usage = {
        'schema_version': 'readonly_openexr_file_compat_usage_v1',
        'status': 'usage_observation_only',
        'pid': os.getpid(), 'python': sys.executable,
        'python_version': sys.version,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'openexr_version': version,
        'openexr_module': str(Path(OpenEXR.__file__).resolve()),
        'openexr_module_sha256': hashlib.sha256(Path(OpenEXR.__file__).read_bytes()).hexdigest(),
        'shim_path': str(shim_path),
        'shim_sha256': hashlib.sha256(shim_path.read_bytes()).hexdigest(),
        'backend_api': 'OpenEXR.InputFile.channel with Imath.PixelType.FLOAT',
        'files_opened': 0, 'channels_decoded': 0, 'first_paths': [],
        'native_File_originally_present': False,
        'version_string_modified': False,
    }

    class ReadOnlyFile:
        def __init__(self, filename, *, separate_channels):
            if separate_channels is not True:
                raise TypeError('Only separate_channels=True is supported by this task-only adapter')
            self._file = OpenEXR.InputFile(str(filename))
            self._closed = False
            self._channels = None
            try:
                self._header = self._file.header()
                window = self._header['dataWindow']
                self._shape = (window.max.y - window.min.y + 1,
                               window.max.x - window.min.x + 1)
                if min(self._shape) <= 0:
                    raise ValueError('Invalid EXR data window')
            except BaseException:
                self._file.close()
                self._closed = True
                raise
            usage['files_opened'] += 1
            if len(usage['first_paths']) < 8:
                usage['first_paths'].append(str(Path(filename).resolve()))

        def __enter__(self):
            if self._closed:
                raise RuntimeError('File already closed')
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._file.close()
            self._closed = True
            return False

        def channels(self):
            if self._closed:
                raise RuntimeError('File already closed')
            if self._channels is None:
                result = {}
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                for name in self._header['channels']:
                    raw = self._file.channel(name, pixel_type)
                    array = np.frombuffer(raw, dtype=np.float32).reshape(self._shape).copy()
                    result[name] = SimpleNamespace(pixels=array)
                    usage['channels_decoded'] += 1
                self._channels = result
            return self._channels

    OpenEXR.File = ReadOnlyFile
    OpenEXR._transparent_depth_file_compat = usage

    def record_usage():
        target = os.environ.get('TD_OPENEXR_COMPAT_AUDIT_DIR')
        if not target:
            return
        root = Path(target)
        root.mkdir(parents=True, exist_ok=True)
        destination = root / ('usage_pid%d.json' % os.getpid())
        temporary = destination.with_suffix('.json.tmp')
        usage['observed_at_exit_unix'] = time.time()
        temporary.write_text(json.dumps(usage, indent=2, sort_keys=True) + '\n')
        temporary.replace(destination)

    atexit.register(record_usage)
