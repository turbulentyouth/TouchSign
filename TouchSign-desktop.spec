"""Single embedded WebView2 application: local files + stdio, no web server."""
from pathlib import Path
import os
from importlib.metadata import distribution
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH)
assets_root = Path(os.environ.get('TOUCHSIGN_ASSETS_DIR', str(root / 'assets')))
datas = [
    (str(assets_root), 'assets'),
    (str(root / 'gui/desktop.html'), 'gui'),
    (str(root / 'gui/desktop.js'), 'gui'),
    (str(root / 'gui/desktop_acceptance.js'), 'gui'),
    (str(root / 'gui/style.css'), 'gui'),
    (str(root / 'config/raw/config.yaml'), 'config'),
    (str(root / 'models/raw/best.onnx'), 'models/raw'),
    (str(root / 'models/raw/best.json'), 'models/raw'),
    (str(root / 'vendor/manifest.json'), 'vendor'),
    (str(root / 'vendor/README.md'), 'vendor'),
]
native_root = root / '.cache/native-window'
for name in ('WebView2Loader.dll',):
    if not (native_root / name).is_file():
        raise RuntimeError(f'Missing native window dependency: {name}; run tools/build_desktop.ps1')
    datas.append((str(native_root / name), 'native-window'))
binaries = collect_dynamic_libs('onnxruntime') + collect_dynamic_libs('agx_cando')
for package in ('python-can', 'python-can-agx-cando', 'pyagxarm', 'wrapt'):
    datas += copy_metadata(package)
webview_dist = distribution('pywebview')
for entry in webview_dist.files or []:
    if str(entry).endswith('.dist-info/LICENSE'):
        datas.append((str(webview_dist.locate_file(entry)), 'vendor/pywebview'))
robot_imports = collect_submodules('pyAgxArm', filter=lambda name: '.demos' not in name and '.scripts' not in name)
robot_imports += collect_submodules('agx_cando')

analysis = Analysis([str(root / 'desktop_main.py')], pathex=[str(root)],
    binaries=binaries, datas=datas,
    hiddenimports=['onnxruntime', 'onnxruntime.capi.onnxruntime_pybind11_state'] + robot_imports,
    excludes=['nicegui', 'webview', 'pythonnet', 'clr', 'clr_loader', 'uvicorn', 'fastapi',
              'starlette', 'socketio', 'engineio', 'websockets', 'aiohttp', 'gui.app',
              'gui.runtime', 'gui.browser_launcher', 'gui.desktop', 'tkinter',
              'torch', 'torchvision', 'torchaudio', 'ultralytics', 'onnx', 'training', 'tests',
              'matplotlib', 'pandas', 'scipy', 'IPython', 'tensorboard'],
    noarchive=False)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, analysis.binaries, analysis.datas, [], name='TouchSign-Desktop',
          debug=False, strip=False, upx=False, console=False,
          runtime_tmpdir=r'%LOCALAPPDATA%\..\LocalLow\TouchSign\unpack')
