"""Assemble an offline Windows folder from official embedded Python and locked wheels.

Run on the build host, never on the target PC. Collected images stay outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def locked_wheels(cache):
    requirements = [line.strip().split('==') for line in
                    (ROOT / 'requirements-windows-portable-lock.txt').read_text().splitlines()
                    if line.strip() and not line.startswith('#')]
    available = {}
    for wheel in (cache / 'windows-wheels').glob('*.whl'):
        name, version, _ = wheel.name.split('-', 2)
        available.setdefault((name.lower().replace('_', '-'), version), []).append(wheel)
    selected = []
    for name, version in requirements:
        matches = available.get((name.lower().replace('_', '-'), version), [])
        if len(matches) != 1:
            raise ValueError(f'锁定依赖缺失或有多个候选：{name}=={version}')
        selected.extend(matches)
    return selected


def unpack_wheel(wheel, site):
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if item.is_dir():
                continue
            parts = path.parts
            if parts[0].endswith('.data'):
                if len(parts) < 3 or parts[1] not in ('purelib', 'platlib'):
                    continue  # CLI launchers/headers are unused; call Python modules directly.
                path = Path(*parts[2:])
            target = (site / path).resolve()
            if not target.is_relative_to(site.resolve()):
                raise ValueError('wheel 路径越界')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))


def copy_sources(target):
    for folder in ('core', 'gui', 'robot', 'training', 'assets', 'config'):
        shutil.copytree(ROOT / folder, target / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('portable.py', 'desktop_main.py', 'collect_manual.py', 'train_easy.py', 'train.py',
                 'requirements-windows-portable-lock.txt'):
        shutil.copy2(ROOT / name, target / name)
    for name in ('LICENSE', 'LICENSE.md', 'LICENSE.txt'):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, target / name)
    shutil.copytree(ROOT / 'models/raw', target / 'models/raw')
    font_dir = target / 'assets/fonts'
    font_dir.mkdir(exist_ok=True)
    shutil.copy2('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', font_dir)
    shutil.copy2('/usr/share/doc/fonts-noto-cjk/copyright', font_dir / 'LICENSE.txt')


def seed_data(run, target):
    from training.portable import load_snapshot, save_snapshot
    rows = []
    for part in ('train', 'val'):
        index = run / 'dataset' / part / 'index.jsonl'
        rows.extend(json.loads(line) for line in index.read_text(encoding='utf-8').splitlines() if line.strip())
    save_snapshot(rows, target / 'data/seed')
    reference = target / 'data/seed/reference'
    reference.mkdir()
    shutil.copy2(run / 'dataset/split.json', reference / 'split.json')
    for part in ('train', 'val'):
        directory = reference / part
        directory.mkdir()
        original = [json.loads(line) for line in (run / 'dataset' / part / 'index.jsonl').read_text(encoding='utf-8').splitlines()]
        (directory / 'index.jsonl').write_text(''.join(json.dumps(
            {k: row[k] for k in ('label', 'sha256', 'session_id', 'episode_id')}) + '\n' for row in original), encoding='utf-8')
    comparison = json.loads((run / 'comparison.json').read_text(encoding='utf-8'))
    external = [row for row in comparison['rows'] if row['scope'] == 'external_rooster']
    save_snapshot(external, target / 'data/regression')
    verified = load_snapshot(target / 'data/seed')
    checks = load_snapshot(target / 'data/regression')
    assert len(verified) == len(rows) and len(checks) == len(external)
    assert not {r['sha256'] for r in rows} & {r['sha256'] for r in checks}
    return {'seed': len(rows), 'regression': len(checks)}


def launchers(target):
    actions = [('01_人工采集', 'collect'), ('02_一键训练', 'train'), ('03_识别与纠错', 'recognize'),
               ('04_回退上次模型', 'rollback'), ('05_运行自检', 'check')]
    for name, action in actions:
        content = ('@echo off\nsetlocal\nchcp 65001 >nul\npushd "%~dp0"\n'
                   'if not exist "runtime\\python.exe" (\n'
                   '  echo Missing runtime. Please extract or copy the entire folder first.\n'
                   '  pause\n  exit /b 1\n)\n'
                   f'"runtime\\python.exe" -X utf8 -u "portable.py" {action}\n'
                   'set "TOUCHSIGN_EXIT=%ERRORLEVEL%"\npopd\npause\nexit /b %TOUCHSIGN_EXIT%\n')
        (target / f'{name}.bat').write_bytes(content.replace('\n', '\r\n').encode('utf-8'))
    shutil.copy2(ROOT / 'docs/windows-portable-quickstart.txt', target / '00_使用说明.txt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/TouchSign-Windows')
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/corrections_final_raw_20261126_121349')
    parser.add_argument('--cache', type=Path, default=ROOT / '.cache')
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    copy_sources(target)
    print('打包模型、训练原图与固定划分…', flush=True)
    counts = seed_data(args.run, target)
    runtime = target / 'runtime'
    with zipfile.ZipFile(args.cache / 'python-3.12.10-embed-amd64.zip') as archive:
        archive.extractall(runtime)
    (runtime / 'python312._pth').write_text('python312.zip\n.\n..\nLib/site-packages\nimport site\n', encoding='utf-8')
    wheels = locked_wheels(args.cache)
    if not wheels:
        raise ValueError('缺少 Windows wheelhouse')
    for wheel in wheels:
        if not (wheel.name.endswith('win_amd64.whl') or wheel.name.endswith('none-any.whl')):
            raise ValueError(f'不是 Windows x64 或纯 Python wheel：{wheel.name}')
        unpack_wheel(wheel, runtime / 'Lib/site-packages')
    for dll in (args.cache / 'windows-crt').glob('*.dll_amd64'):
        shutil.copy2(dll, runtime / dll.name.removesuffix('_amd64'))
    if not (runtime / 'msvcp140.dll').exists():
        raise ValueError('缺少 Microsoft 应用本地 C++ 运行库')
    licenses = target / 'licenses'
    licenses.mkdir(exist_ok=True)
    for source, name in (('u4', 'Microsoft-VC-Runtime-en.rtf'), ('u28', 'Microsoft-VC-Runtime-zh.rtf')):
        shutil.copy2(args.cache / 'vcredist-all' / source, licenses / name)
    # Native compilation SDKs are unnecessary for this training/inference workflow.
    shutil.rmtree(runtime / 'Lib/site-packages/torch/include', ignore_errors=True)
    for library in (runtime / 'Lib/site-packages').rglob('*.lib'):
        library.unlink()
    launchers(target)
    provenance = {'python_url': 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip',
                  'python_sha256': sha(args.cache / 'python-3.12.10-embed-amd64.zip'),
                  'crt_url': 'https://aka.ms/vs/17/release/vc_redist.x64.exe',
                  'crt_sha256': sha(args.cache / 'vc_redist.x64.exe'),
                  'wheels': {w.name: sha(w) for w in wheels}, 'data': counts,
                  'target': 'Windows 10/11 x64, CPU; local browser', 'windows_hardware_tested': False}
    (target / 'BUILD.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    inventory = [f'{sha(path)}  {path.relative_to(target).as_posix()}\n'
                 for path in sorted(target.rglob('*')) if path.is_file()]
    (target / 'SHA256SUMS.txt').write_text(''.join(inventory), encoding='utf-8')
    print(f'完成：{target}，{counts}，{len(wheels)} 个依赖包', flush=True)


if __name__ == '__main__':
    main()
