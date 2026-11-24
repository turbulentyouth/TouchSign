param([string]$Python = '.venv-runtime\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$previousBuildCache = $env:PYINSTALLER_CONFIG_DIR
$env:PYINSTALLER_CONFIG_DIR = Join-Path $projectRoot '.cache\pyinstaller'
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw 'Create a Python 3.12 virtual environment and install requirements-build.txt first.'
    }
    & $Python -c @'
from importlib import metadata, util
from pathlib import Path
import json
import os
import wave
from core.zodiac import ZODIAC_KEYS

assets_root = Path(os.environ.get('TOUCHSIGN_ASSETS_DIR', 'assets'))
manifest = json.loads((assets_root / 'speech/manifest.json').read_text(encoding='utf-8'))
if set(manifest) != set(ZODIAC_KEYS):
    raise ValueError('Speech manifest must contain all twelve zodiac entries')
if not (assets_root / 'speech-test.js').is_file():
    raise FileNotFoundError('Missing assets/speech-test.js')
for key in ZODIAC_KEYS:
    with wave.open(str(assets_root / 'speech' / f'{key}.wav'), 'rb') as audio:
        if audio.getnframes() == 0:
            raise ValueError(f'Empty speech file: {key}')
print('All twelve speech recordings and playback assets verified')
import hashlib
for entry in json.loads(Path('vendor/manifest.json').read_text(encoding='utf-8')):
    path = Path('vendor/wheels') / entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'Incorrect SDK dependency: {path}')
from can.interface import _get_class_for_interface
from agx_cando.dll import load_cando_dll
from agx_cando.native import _load_receive_api
from pyAgxArm import create_agx_arm_config, AgxArmFactory
assert _get_class_for_interface('agx_cando').__name__ == 'AgxCandoBus'
try:
    load_cando_dll()
    _load_receive_api()
    print('Native CAN DLLs loaded; receiver ABI 2 verified')
except OSError as exc:
    if getattr(exc, 'winerror', None) != 4551:
        raise
    print('LIMITATION: Windows application control blocks the upstream CAN DLL. '
          'Building offline-preview package; hardware loading remains unverified/blocked on this PC.')
for profile in ('default', 'v111', 'v112', 'v120', 'v121'):
    # Construct only: never connect or enable a physical arm during packaging.
    AgxArmFactory.create_arm(create_agx_arm_config(robot='nero', firmeware_version=profile,
                                                  interface='agx_cando', channel='0'))
print('SDK imports and five firmware drivers verified (no hardware opened)')

bad = [name for name in ('torch', 'torchvision', 'torchaudio', 'ultralytics') if util.find_spec(name)]
bad += [dist.metadata['Name'] for dist in metadata.distributions()
        if dist.metadata['Name'].lower().startswith(('onnxruntime-gpu', 'nvidia-'))]
if bad:
    raise RuntimeError(f'Use a clean CPU environment. Found {bad}')

from gui.runtime import prepare_runtime
prepare_runtime()
from core.config import load_config, resolve_path
from core.inference import OnnxClassifier, RAW_INPUT_KIND
settings = load_config('config/raw/config.yaml').recognition
model = OnnxClassifier(resolve_path(settings.model_path), preview=settings.preview)
if model.input_kind != RAW_INPUT_KIND:
    raise ValueError('Expected a raw-frame classifier')
print('Raw ONNX, manifest and CPU inference session verified')
'@
    if ($LASTEXITCODE -ne 0) { throw 'Model or build environment check failed.' }

    & $Python -c "from gui.runtime import prepare_runtime; prepare_runtime(); import PyInstaller.__main__; PyInstaller.__main__.run(['--noconfirm', '--clean', 'TouchSign.spec'])"
    if ($LASTEXITCODE -ne 0) { throw 'EXE build failed.' }

    & $Python -c @'
from pathlib import Path
import os
from PyInstaller.archive.readers import CArchiveReader, pkg_archive_contents
exe = Path('dist/TouchSign.exe')
archive = CArchiveReader(exe)
expected_tmpdir = r'pyi-runtime-tmpdir %LOCALAPPDATA%\..\LocalLow\TouchSign\unpack'
if expected_tmpdir not in archive.options:
    raise ValueError('Missing per-user LocalLow extraction path')
import agx_cando
plugin = Path(agx_cando.__file__).parent
for name in ('cando.dll', 'agx_receive.dll'):
    relative = Path('agx_cando/bin/x64') / name
    if archive.extract(str(relative)) != (plugin / 'bin/x64' / name).read_bytes():
        raise ValueError(f'Incorrect embedded CAN DLL: {relative}')
for path in (Path('models/raw/best.onnx'), Path('models/raw/best.json')):
    if archive.extract(str(path)) != path.read_bytes():
        raise ValueError(f'Incorrect embedded model file {path}')
if archive.extract(str(Path('config/config.yaml'))) != Path('config/raw/config.yaml').read_bytes():
    raise ValueError('Incorrect embedded startup profile')
assets_root = Path(os.environ.get('TOUCHSIGN_ASSETS_DIR', 'assets'))
for path in (assets_root / 'speech').glob('*'):
    embedded = Path('assets/speech') / path.name
    if path.is_file() and archive.extract(str(embedded)) != path.read_bytes():
        raise ValueError(f'Incorrect embedded speech file {path}')
for entry in pkg_archive_contents(exe):
    path = Path(entry.lower())
    if (path.parts[0].split('.')[0] in {'torch', 'torchvision', 'torchaudio', 'ultralytics', 'training', 'onnx', 'nvidia'}
            or path.name.startswith(('cublas', 'cudnn', 'cufft', 'curand', 'cusolver', 'cusparse', 'nvrtc',
                                      'onnxruntime_providers_cuda', 'onnxruntime_providers_tensorrt'))):
        raise ValueError(f'Training or CUDA dependency in EXE: {entry}')
print(f'Archive verified: {exe.resolve()} ({exe.stat().st_size / 1024**2:.1f} MiB)')
'@
    if ($LASTEXITCODE -ne 0) { throw 'Single-file archive verification failed.' }
    # The one-file bootloader extracts into per-user LocalLow/TouchSign.
    # Do not depend on changing an inherited integrity label during packaging.
    & $Python tools/package_release.py
    if ($LASTEXITCODE -ne 0) { throw 'Single-EXE ZIP verification failed.' }
    Write-Output 'Release: dist\TouchSign-NERO-SingleFile.zip (extract using Explorer into a normal user directory)'
} finally {
    $env:PYINSTALLER_CONFIG_DIR = $previousBuildCache
    Pop-Location
}
