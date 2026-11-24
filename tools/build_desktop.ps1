param([string]$Python = '.venv-runtime\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$previousBuildCache = $env:PYINSTALLER_CONFIG_DIR
$env:PYINSTALLER_CONFIG_DIR = Join-Path $projectRoot '.cache\pyinstaller-desktop'
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Missing runtime Python environment.' }
    if (-not (Test-Path -LiteralPath 'desktop_main.py')) { throw 'Missing native desktop entry point.' }
    & $Python -c @'
from importlib.util import find_spec
from pathlib import Path
import shutil
source = Path(find_spec('webview').origin).parent / 'lib/runtimes/win-x64/native/WebView2Loader.dll'
destination = Path('.cache/native-window/WebView2Loader.dll')
destination.parent.mkdir(parents=True, exist_ok=True)
if not destination.is_file() or destination.read_bytes() != source.read_bytes():
    shutil.copy2(source, destination)
assert destination.read_bytes() == source.read_bytes()
print('Microsoft WebView2 loader verified; no custom executable compilation')
'@
    if ($LASTEXITCODE -ne 0) { throw 'WebView2 loader preparation failed.' }
    & $Python -c @'
from pathlib import Path
import hashlib
import json
import os
import wave
from core.zodiac import ZODIAC_KEYS
from core.inference import OnnxClassifier, RAW_INPUT_KIND

model = OnnxClassifier(Path('models/raw/best.onnx'), preview=True)
assert model.input_kind == RAW_INPUT_KIND
print('CPU ONNX model and manifest verified')
assets = Path(os.environ.get('TOUCHSIGN_ASSETS_DIR', 'assets'))
for key in ZODIAC_KEYS:
    with wave.open(str(assets / 'speech' / f'{key}.wav'), 'rb') as audio:
        assert audio.getnframes() > 0
print('Twelve speech recordings verified')
for entry in json.loads(Path('vendor/manifest.json').read_text(encoding='utf-8')):
    assert hashlib.sha256((Path('vendor/wheels') / entry['file']).read_bytes()).hexdigest() == entry['sha256']
from can.interface import _get_class_for_interface
from pyAgxArm import AgxArmFactory, create_agx_arm_config
assert _get_class_for_interface('agx_cando').__name__ == 'AgxCandoBus'
for profile in ('default', 'v111', 'v112', 'v120', 'v121'):
    AgxArmFactory.create_arm(create_agx_arm_config(robot='nero', firmeware_version=profile, interface='agx_cando', channel='0'))
print('Pinned CAN dependencies and five firmware profiles verified; no hardware opened')
inputs = [Path('desktop_main.py'), Path('TouchSign-desktop.spec')]
for directory in ('core', 'robot'):
    inputs.extend(Path(directory).glob('*.py'))
inputs.extend(Path('gui') / name for name in ('embedded.py', 'win32_webview.py', 'desktop_backend.py',
                                             'view_models.py', 'desktop.html', 'desktop.js', 'desktop_acceptance.js', 'style.css'))
Path('.cache/desktop-build-inputs.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}), encoding='utf-8')
'@
    if ($LASTEXITCODE -ne 0) { throw 'Native desktop preflight failed.' }

    & $Python -m PyInstaller --noconfirm --clean TouchSign-desktop.spec
    if ($LASTEXITCODE -ne 0) { throw 'Native desktop EXE build failed.' }

    & $Python -c @'
from pathlib import Path
import os
import hashlib
import json
from PyInstaller.archive.readers import CArchiveReader, pkg_archive_contents
import agx_cando
exe = Path('dist/TouchSign-Desktop.exe')
archive = CArchiveReader(exe)
for name, digest in json.loads(Path('.cache/desktop-build-inputs.json').read_text(encoding='utf-8')).items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
        raise ValueError(f'Source changed during packaging; rebuild required: {name}')
assert r'pyi-runtime-tmpdir %LOCALAPPDATA%\..\LocalLow\TouchSign\unpack' in archive.options
for name in ('models/raw/best.onnx', 'models/raw/best.json'):
    assert archive.extract(str(Path(name))) == Path(name).read_bytes(), name
assert archive.extract(str(Path('config/config.yaml'))) == Path('config/raw/config.yaml').read_bytes()
assets = Path(os.environ.get('TOUCHSIGN_ASSETS_DIR', 'assets'))
for source in (assets / 'speech').glob('*'):
    if source.is_file():
        assert archive.extract(str(Path('assets/speech') / source.name)) == source.read_bytes(), source
plugin = Path(agx_cando.__file__).parent
for name in ('cando.dll', 'agx_receive.dll'):
    assert archive.extract(str(Path('agx_cando/bin/x64') / name)) == (plugin / 'bin/x64' / name).read_bytes(), name
entries = pkg_archive_contents(exe)
for name in ('desktop.html', 'desktop.js', 'desktop_acceptance.js', 'style.css'):
    assert archive.extract(str(Path('gui') / name)) == (Path('gui') / name).read_bytes(), name
for name in ('WebView2Loader.dll',):
    assert archive.extract(str(Path('native-window') / name)) == (Path('.cache/native-window') / name).read_bytes(), name
for name in entries:
    parts = name.lower().replace('\\', '/').split('/')
    stem = parts[0].split('.')[0]
    if stem in {'nicegui', 'webview', 'pythonnet', 'clr', 'clr_loader', 'uvicorn', 'fastapi',
                'starlette', 'socketio', 'engineio', 'websockets', 'aiohttp',
                'torch', 'torchvision', 'torchaudio', 'ultralytics', 'onnx', 'training', 'nvidia'}:
        raise ValueError(f'Forbidden browser/server/training dependency: {name}')
    if parts[-1] == 'python.runtime.dll':
        raise ValueError('Python.NET runtime unexpectedly bundled')
    if parts[-1] in {'touchsignwindow.exe', 'microsoft.web.webview2.core.dll', 'microsoft.web.webview2.winforms.dll'}:
        raise ValueError(f'Obsolete .NET window host unexpectedly bundled: {name}')
print(f'Native archive verified: {exe.resolve()} ({exe.stat().st_size / 1024**2:.1f} MiB)')
'@
    if ($LASTEXITCODE -ne 0) { throw 'Native desktop archive verification failed.' }
    $testRoot = Join-Path $projectRoot '.cache\desktop-frozen-profile'
    New-Item -ItemType Directory -Force -Path (Join-Path $testRoot 'Local') | Out-Null
    $originalLocal = $env:LOCALAPPDATA
    try {
        $env:LOCALAPPDATA = Join-Path $testRoot 'Local'
        foreach ($test in @(
            @{ Flag = '--self-test-report'; File = '.cache\desktop-frozen-native-acceptance.json' },
            @{ Flag = '--diagnostics-report'; File = '.cache\desktop-frozen-diagnostics.json' }
        )) {
            $reportPath = Join-Path $projectRoot $test.File
            if (Test-Path -LiteralPath $reportPath) { Remove-Item -LiteralPath $reportPath -ErrorAction Stop }
            try {
                $testProcess = Start-Process -FilePath (Join-Path $projectRoot 'dist\TouchSign-Desktop.exe') -ArgumentList $test.Flag, $reportPath -WindowStyle Hidden -PassThru
            } catch {
                [ordered]@{ passed = $false; stage = 'windows_process_creation'; error = $_.Exception.Message;
                    hardware_opened = $false; executable_sha256 = (Get-FileHash -LiteralPath (Join-Path $projectRoot 'dist\TouchSign-Desktop.exe') -Algorithm SHA256).Hash
                } | ConvertTo-Json | Set-Content -LiteralPath $reportPath -Encoding UTF8
                throw
            }
            if (-not $testProcess.WaitForExit(60000)) {
                Stop-Process -Id $testProcess.Id
                throw "Frozen acceptance timed out: $($test.Flag)"
            }
            if ($testProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $reportPath)) { throw "Frozen acceptance failed: $($test.Flag)" }
            $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($test.Flag -eq '--self-test-report') {
                if (-not ($report.passed -and $report.exit_button_clicked -and $report.backend_thread_stopped)) { throw 'GUI acceptance did not finish all actions and exit cleanly.' }
            } elseif (-not $report.offline_checks_passed) { throw 'Frozen diagnostics did not pass.' }
            Write-Output "Frozen acceptance passed: $reportPath"
        }
    } finally {
        $env:LOCALAPPDATA = $originalLocal
    }
    Write-Output 'Ready: dist\TouchSign-Desktop.exe'
} finally {
    $env:PYINSTALLER_CONFIG_DIR = $previousBuildCache
    Pop-Location
}
