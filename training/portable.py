"""Relocatable, verified raw-image snapshots for incremental desktop training."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from core.inference import CLASS_NAMES, RAW_INPUT_KIND

FORMAT = 'touchsign_portable_raw_v1'


def pixel_hash(frame):
    return hashlib.sha256(np.asarray(frame.shape, dtype='<u8').tobytes() + frame.tobytes()).hexdigest()


def inside(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'便携数据路径越界：{relative}')
    return path


def load_snapshot(root):
    from training.data import read_image
    root = Path(root).resolve()
    document = json.loads((root / 'snapshot.json').read_text(encoding='utf-8'))
    if document.get('format') != FORMAT or document.get('input_kind') != RAW_INPUT_KIND:
        raise ValueError('便携数据快照格式不匹配')
    samples, seen = [], set()
    for row in document['samples']:
        path = inside(root, row['image'])
        if row['label'] not in CLASS_NAMES or not row.get('session_id') or not row.get('episode_id'):
            raise ValueError('便携数据标签或按压分组无效')
        if row.get('synthetic'):
            raise ValueError('便携续训不接受合成数据')
        digest = pixel_hash(read_image(path))
        if digest != row['sha256'] or digest in seen:
            raise ValueError(f'数据已修改或存在重复图像：{path}')
        seen.add(digest)
        samples.append({**row, 'path': path, 'input_kind': RAW_INPUT_KIND})
    if not samples:
        raise ValueError('便携数据快照为空')
    return samples


def save_snapshot(rows, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    saved = []
    for row in rows:
        relative = f"images/{row['label']}/{row['sha256']}.png"
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row['path'], path)
        saved.append({key: row[key] for key in ('label', 'session_id', 'episode_id', 'sha256')} |
                     {'image': relative, 'synthetic': False})
    (destination / 'snapshot.json').write_text(json.dumps(
        {'format': FORMAT, 'input_kind': RAW_INPUT_KIND, 'samples': saved},
        ensure_ascii=False, indent=2), encoding='utf-8')


def merge_new(base, batches, protected=()):
    """Retain old press groups; do not turn a duplicate holdout into training data."""
    known = {row['sha256']: row for row in base}
    protected_hashes = {row['sha256'] for row in protected}
    protected_groups = {(row['session_id'], row['episode_id']) for row in protected}
    added = []
    for row in batches:
        digest = row['sha256']
        if digest in protected_hashes or (row['session_id'], row['episode_id']) in protected_groups:
            raise ValueError('新增采集数据与独立回归检查图像/按压重复，请从采集目录移除该副本')
        if digest in known:
            old = known[digest]
            if (old['label'], old['session_id']) != (row['label'], row['session_id']):
                raise ValueError('同一图像出现在不同标签或会话，停止训练以避免标签冲突/数据泄漏')
            continue
        known[digest] = row
        added.append(row)
    return [*base, *added], added
