"""Atomic model selection shared by the portable recognizer and trainer."""
from __future__ import annotations

import json
import os
from pathlib import Path

from core.inference import file_sha256

FACTORY = {'model_dir': 'models/raw', 'snapshot': 'data/seed', 'reference': 'data/seed/reference'}


def local_path(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('模型记录路径必须位于便携目录内')
    return path


def validate_record(root, record):
    directory = local_path(root, record['model_dir'])
    manifest = json.loads((directory / 'best.json').read_text(encoding='utf-8'))
    if file_sha256(directory / 'best.onnx') != manifest['sha256']:
        raise ValueError('模型文件与校准文件不匹配')
    weights = directory / 'best.pt'
    if not weights.is_file():
        raise ValueError('缺少续训权重 best.pt')
    if record.get('weights_sha256') and file_sha256(weights) != record['weights_sha256']:
        raise ValueError('续训权重校验失败')
    for key in ('snapshot', 'reference'):
        if not local_path(root, record[key]).is_dir():
            raise ValueError('缺少训练数据或固定验证划分')
    return record


def selected(root):
    pointer = root / 'models/active.json'
    record = json.loads(pointer.read_text(encoding='utf-8')) if pointer.exists() else dict(FACTORY)
    if record['model_dir'] == FACTORY['model_dir']:
        release = root / 'models/raw/release.json'
        if release.exists():
            record['weights_sha256'] = json.loads(release.read_text(encoding='utf-8'))['files']['best.pt']
    return validate_record(root, record)


def activate(root, record, previous):
    validate_record(root, record)
    document = {**record, 'previous': {k: v for k, v in previous.items() if k != 'previous'}}
    target = root / 'models/active.json'
    temp = target.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def rollback(root):
    current = selected(root)
    previous = current.get('previous')
    if previous is None:
        return False
    activate(root, previous, current)
    return True
