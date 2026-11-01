# Build only in the clean requirements-build.txt environment.
from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(SPECPATH)
datas = collect_data_files('nicegui')
datas += [(str(root/'assets'), 'assets'), (str(root/'gui'/'style.css'), 'gui'),
          (str(root/'config'/'raw'/'config.yaml'), 'config'),
          (str(root/'models'/'raw'/'best.onnx'), 'models/raw'),
          (str(root/'models'/'raw'/'best.json'), 'models/raw')]
binaries = collect_dynamic_libs('onnxruntime')
# Preserve the existing optional SSL compatibility mechanism in frozen builds.
# Never collect arbitrary system directories; only the explicit override or the
# project's documented two-DLL runtime directory is eligible.
openssl_dir = Path(os.environ.get('TOUCHSIGN_OPENSSL_DIR', str(root/'runtime'/'openssl')))
openssl_names = ('libcrypto-3-x64.dll', 'libssl-3-x64.dll')
if openssl_dir.is_dir():
    if not all((openssl_dir/name).is_file() for name in openssl_names):
        raise RuntimeError('OpenSSL 兼容目录必须同时提供 libcrypto-3-x64.dll 和 libssl-3-x64.dll')
    binaries += [(str(openssl_dir/name), 'runtime/openssl') for name in openssl_names]
a = Analysis([str(root/'main.py')], pathex=[str(root)],
    binaries=binaries, datas=datas,
    hiddenimports=['onnxruntime', 'onnxruntime.capi.onnxruntime_pybind11_state'],
    excludes=['torch', 'torchvision', 'torchaudio', 'ultralytics', 'onnx', 'training',
              'tests', 'matplotlib', 'pandas', 'scipy', 'IPython', 'tensorboard'],
    noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='TouchSign',
          debug=False, strip=False, upx=False, console=False)
