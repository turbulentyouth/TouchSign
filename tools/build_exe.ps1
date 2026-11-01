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
from PyInstaller.archive.readers import CArchiveReader, pkg_archive_contents
exe = Path('dist/TouchSign.exe')
archive = CArchiveReader(exe)
for path in (Path('models/raw/best.onnx'), Path('models/raw/best.json')):
    if archive.extract(str(path)) != path.read_bytes():
        raise ValueError(f'Incorrect embedded model file {path}')
if archive.extract(str(Path('config/config.yaml'))) != Path('config/raw/config.yaml').read_bytes():
    raise ValueError('Incorrect embedded startup profile')
for entry in pkg_archive_contents(exe):
    path = Path(entry.lower())
    if (path.parts[0].split('.')[0] in {'torch', 'torchvision', 'torchaudio', 'ultralytics', 'training', 'onnx', 'nvidia'}
            or path.name.startswith(('cublas', 'cudnn', 'cufft', 'curand', 'cusolver', 'cusparse', 'nvrtc',
                                      'onnxruntime_providers_cuda', 'onnxruntime_providers_tensorrt'))):
        raise ValueError(f'Training or CUDA dependency in EXE: {entry}')
print(f'Ready: {exe.resolve()} ({exe.stat().st_size / 1024**2:.1f} MiB)')
'@
    if ($LASTEXITCODE -ne 0) { throw 'Single-file archive verification failed.' }
} finally {
    $env:PYINSTALLER_CONFIG_DIR = $previousBuildCache
    Pop-Location
}
