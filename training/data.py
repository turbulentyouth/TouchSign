"""Validate collection manifests and split by recording session or contact episode."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from core.inference import CLASS_NAMES, INPUT_KIND, RAW_INPUT_KIND
from core.manual_collection import FORMAT as MANUAL_FORMAT


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or not image.size:
        raise ValueError(f'无法解码采集图像：{path}')
    return image


def _manual_row(row: dict, manifest: Path, root: Path, grouping: str) -> dict:
    """Normalize manual paths without inventing a baseline or contact measurement."""
    if row.get('format') != MANUAL_FORMAT or row.get('label_source') != 'manual':
        raise ValueError('手工采集格式或标签来源无效')
    if row.get('rotation_degrees') != 180:
        raise ValueError('手工图像必须已按 180° 保存，训练不会再次旋转')
    image = (manifest.parent / row['image']).resolve()
    if image.parent != (manifest.parent / row['label']).resolve():
        raise ValueError('手工图像目录与标签不一致')
    relative = image.relative_to(root).as_posix()
    # A frame is an independent press only when the operator explicitly says so.
    # Unloaded frames from one session remain together, even in frame mode.
    episode = f'manual-{image.stem}' if grouping == 'frame' and row['label'] != 'no_contact' else 'manual-session'
    correction = row.get('correction')
    if correction is not None:
        from core.corrections import CORRECTION_FORMAT
        if (not isinstance(correction, dict) or correction.get('format') != CORRECTION_FORMAT
                or not isinstance(correction.get('episode_id'), str) or not correction['episode_id']):
            raise ValueError('纠错样本缺少有效按压分组')
        if grouping == 'frame':
            # Several saved positions during one held press are still one group.
            episode = 'correction-' + correction['episode_id']
    return {**row, 'image': relative, 'raw_image': relative,
            'session_id': manifest.parent.name, 'episode_id': episode, 'input_kind': RAW_INPUT_KIND}


def read_samples(root: str | Path, *, input_kind: str = INPUT_KIND,
                 manual_grouping: str = 'session') -> list[dict]:
    """Keep the collection metadata; path and sha256 identify the selected model input."""
    if input_kind not in (INPUT_KIND, RAW_INPUT_KIND):
        raise ValueError(f'不支持的模型输入：{input_kind}')
    if manual_grouping not in ('session', 'frame'):
        raise ValueError('manual_grouping 必须为 session 或 frame')
    image_field = 'raw_image' if input_kind == RAW_INPUT_KIND else 'image'
    root = Path(root).resolve()
    manifests = sorted([*root.glob('*/manifest.jsonl'), *root.glob('*/samples.jsonl')])
    if not manifests:
        raise ValueError(f'没有采集数据：{root}。请先采集十二生肖、未知物体和无接触数据。')
    samples, paths, hashes = [], set(), {}
    for manifest in manifests:
        manual = manifest.name == 'samples.jsonl'
        if manual:
            if input_kind != RAW_INPUT_KIND:
                raise ValueError('手工采集只有原图，请使用 --input raw')
            metadata = json.loads((manifest.parent / 'session.json').read_text(encoding='utf-8'))
            if (metadata.get('format') != MANUAL_FORMAT or metadata.get('rotation_degrees') != 180
                    or metadata.get('input_kind') != RAW_INPUT_KIND
                    or metadata.get('session_id') != manifest.parent.name):
                raise ValueError(f'{manifest.parent}: 手工会话信息无效')
        cached_baseline_path = baseline = None
        for line_no, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if manual:
                    row = _manual_row(row, manifest, root, manual_grouping)
                if input_kind == RAW_INPUT_KIND and not row.get('raw_image'):
                    raise ValueError('清单缺少原图路径 raw_image')
                image = (root / row[image_field]).resolve()
                label, session = row['label'], row['session_id']
                if not image.is_relative_to(manifest.parent.resolve()) or not image.is_file():
                    raise ValueError('图像缺失或路径越界')
                if session != manifest.parent.name or label not in CLASS_NAMES or not row.get('episode_id'):
                    raise ValueError('session_id、label 或 episode_id 无效')
                if not manual and row.get('input_kind') != INPUT_KIND:
                    raise ValueError('采集清单格式不匹配，请用新版采集界面采集')
                if not manual and row.get('contact_detected') is not (label != 'no_contact'):
                    raise ValueError('接触标注不符合类别：生肖和 unknown 必须确有接触，no_contact 必须无接触')
                if image in paths:
                    raise ValueError('清单包含重复图像路径')
                if not manual and input_kind == RAW_INPUT_KIND and row.get('image') and image == (root / row['image']).resolve():
                    raise ValueError('raw_image 与差分图指向同一文件，请核对原图记录')
                frame = read_image(image)
                if manual and list(frame.shape) != row.get('frame_shape'):
                    raise ValueError('手工图像尺寸与采集记录不一致')
                # Hash decoded pixels, not PNG bytes: changing compression or
                # adding metadata must not bypass the cross-session leak check.
                digest = hashlib.sha256(
                    np.asarray(frame.shape, dtype='<u8').tobytes() + frame.tobytes()
                ).hexdigest()
                if digest in hashes and hashes[digest] != (session, label):
                    raise ValueError('同一图像出现在不同 session 或类别，会导致数据泄漏或标签冲突')
                if digest in hashes:
                    continue  # duplicate unchanged frames do not count as more training evidence
                if input_kind == INPUT_KIND and not row.get('synthetic', False):
                    raw_path, baseline_path = [(root / row[field]).resolve() for field in ('raw_image', 'baseline')]
                    if any(not path.is_relative_to(manifest.parent.resolve()) or not path.is_file()
                           for path in (raw_path, baseline_path)):
                        raise ValueError('原始图像或空载基准缺失/路径越界')
                    raw_frame = read_image(raw_path)
                    if baseline_path != cached_baseline_path:
                        baseline = read_image(baseline_path)
                        cached_baseline_path = baseline_path
                    if raw_frame.shape != baseline.shape or frame.shape != raw_frame.shape:
                        raise ValueError('原始图像、差分图、基准图尺寸不一致')
                    if not np.array_equal(frame, cv2.absdiff(raw_frame, baseline)):
                        raise ValueError('image 不是对应 raw_image 与 baseline 的绝对差分图')
                paths.add(image)
                hashes[digest] = (session, label)
                samples.append({**row, 'path': image, 'sha256': digest})
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{manifest}:{line_no}: {exc}') from exc
    if not samples:
        raise ValueError('采集清单没有有效图像')
    return samples


def _validation_groups(groups: dict, *, seed: int, minimum: int, allow_train_only: bool):
    """Select whole groups, aiming for 20% by image count within each class."""
    keys = sorted(groups)
    label_counts = [Counter(row['label'] for row in groups[key]) for key in keys]
    counts = np.array([[counter[label] for label in CLASS_NAMES] for counter in label_counts])
    totals = counts.sum(axis=0)
    support = (counts > 0).sum(axis=0)
    eligible = support >= 2
    present = totals > 0
    if not eligible.any() or (not allow_train_only and not (eligible | ~present).all()):
        return None
    if (totals < minimum * (present.astype(int) + eligible.astype(int))).any():
        return None

    # A class observed in just one episode must stay in training. Redistribute
    # its share of validation across the eligible classes to retain the 80/20 total.
    goal = max(1, round(int(totals.sum()) * 0.2))
    quotas = goal * (totals * eligible) / totals[eligible].sum()
    targets = np.floor(quotas).astype(int)
    remainder = goal - int(targets.sum())
    targets[np.argsort(-(quotas - targets), kind='stable')[:remainder]] += 1
    candidates = [i for i, row in enumerate(counts) if not row[~eligible].any()]
    if not candidates:
        return None

    def score(val):
        shortfall = (np.maximum(minimum * eligible - val, 0).sum()
                     + np.maximum(minimum * present - (totals - val), 0).sum())
        return shortfall, np.abs(val - targets).sum(), abs(val.sum() - goal)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    best_score, best_indices = None, set()
    if len(candidates) <= 14:
        # For a small number of recording sessions we can check every subset.
        for size in range(1, len(candidates) + 1):
            for subset in combinations(candidates, size):
                current = score(counts[list(subset)].sum(axis=0))
                if current == (0, 0, 0):
                    return {keys[i] for i in subset}
                if best_score is None or current < best_score:
                    best_score, best_indices = current, set(subset)
    else:
        # Many episodes can have identical sizes. Randomize ties reproducibly
        # and allow removing groups when that improves class coverage or totals.
        for _ in range(64):
            rng.shuffle(candidates)
            selected = set()
            val = np.zeros_like(totals)
            current = score(val)
            for _ in range(3):
                changed = False
                for idx in candidates:
                    trial = val - counts[idx] if idx in selected else val + counts[idx]
                    trial_score = score(trial)
                    if trial_score < current:
                        selected.symmetric_difference_update((idx,))
                        val, current, changed = trial, trial_score, True
                if not changed:
                    break
            if best_score is None or current < best_score:
                best_score, best_indices = current, selected
            if current == (0, 0, 0):
                break
    if best_score is None or best_score[0] or not best_indices:
        return None
    return {keys[i] for i in best_indices}


def _grouped_split(samples: list[dict], *, seed: int, min_per_class: int,
                   allow_missing_unknown: bool = False):
    if min_per_class < 1:
        raise ValueError('每类的最少图像数必须为正数')
    totals = Counter(s['label'] for s in samples)
    missing = [label for label in CLASS_NAMES if totals[label] < min_per_class
               and not (allow_missing_unknown and label == 'unknown' and totals[label] == 0)]
    if missing:
        raise ValueError(f'训练集每类至少需要 {min_per_class} 张有效图像，数据不足：{missing}')
    for grouping in ('session', 'episode'):
        groups = defaultdict(list)
        for row in samples:
            key = (row['session_id'],) if grouping == 'session' else (row['session_id'], row['episode_id'])
            groups[key].append(row)
        validation = _validation_groups(groups, seed=seed, minimum=min_per_class,
                                        allow_train_only=grouping == 'episode')
        if validation is not None:
            split = {'train': [], 'val': []}
            for key, rows in groups.items():
                split['val' if key in validation else 'train'].extend(rows)
            return split, grouping
    raise ValueError(f'无法按完整采集批次或按压划分训练和验证集。请为不足的类别补采独立按压，'
                     f'有多个按压的类别每侧至少需要 {min_per_class} 张有效图像。')


def split_sessions(samples: list[dict], *, seed: int = 42, min_per_class: int = 1) -> dict[str, list[dict]]:
    """Prefer whole sessions, falling back to whole (session, episode) groups."""
    split, _ = _grouped_split(samples, seed=seed, min_per_class=min_per_class)
    return split


def _reference_split(samples: list[dict], reference: Path, *, seed: int,
                     min_per_class: int, input_kind: str):
    """Keep previously held-out images out of incremental training, by pixel hash.

    New frames in an existing episode inherit that episode's assignment. New
    episodes are split as whole groups; sparse classes may exceed the 20% target
    so that a new acquisition setup has validation coverage wherever possible.
    """
    reference = reference.resolve()
    metadata = json.loads((reference / 'split.json').read_text(encoding='utf-8'))
    if metadata['input_kind'] != input_kind:
        raise ValueError('参考划分的图像类型与本次训练不同')
    by_hash = {row['sha256']: row for row in samples}
    known, assignments = {}, {}
    for name in ('train', 'val'):
        for line in (reference / name / 'index.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            digest = row['sha256']
            if digest in known:
                raise ValueError('参考划分包含重复图像或训练/验证交叉')
            if digest not in by_hash:
                raise ValueError(f'参考划分中的图像已缺失；请核对数据后再续训：{digest}')
            sample = by_hash[digest]
            if row['label'] != sample['label']:
                raise ValueError('参考划分中的标签已改变，不能直接沿用旧权重续训')
            key = sample['session_id'], sample['episode_id']
            if key in assignments and assignments[key] != name:
                raise ValueError('参考划分将同一次按压分到了训练和验证两侧')
            known[digest] = assignments[key] = name
    if set(known.values()) != {'train', 'val'}:
        raise ValueError('参考划分必须同时包含训练和验证样本')
    new_groups = defaultdict(list)
    for row in samples:
        key = row['session_id'], row['episode_id']
        if key not in assignments:
            new_groups[key].append(row)
    validation = (_validation_groups(new_groups, seed=seed, minimum=min_per_class,
                                    allow_train_only=True) if new_groups else None)
    assignments.update({key: 'val' if key in (validation or set()) else 'train'
                        for key in new_groups})
    split = {'train': [], 'val': []}
    for row in samples:
        split[assignments[row['session_id'], row['episode_id']]].append(row)
    details = {'path': str(reference),
               'retained': dict(Counter(known.values())),
               'added': {name: sum(row['sha256'] not in known for row in rows)
                         for name, rows in split.items()}}
    return split, details


def prepare_dataset(raw: str | Path, destination: str | Path, *, seed: int = 42,
                    min_per_class: int = 1, input_kind: str = INPUT_KIND,
                    manual_grouping: str = 'session', allow_missing_unknown: bool = False,
                    reference_split: str | Path | None = None) -> tuple[dict, dict]:
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'输出目录非空，避免混入旧数据：{destination}')
    if allow_missing_unknown and input_kind != RAW_INPUT_KIND:
        raise ValueError('缺少 unknown 的试训仅支持原图模式')
    samples = read_samples(raw, input_kind=input_kind, manual_grouping=manual_grouping)
    split, grouping = _grouped_split(samples, seed=seed, min_per_class=min_per_class,
                                     allow_missing_unknown=allow_missing_unknown)
    reference_details = None
    if reference_split is not None:
        split, reference_details = _reference_split(samples, Path(reference_split), seed=seed,
                                                    min_per_class=min_per_class, input_kind=input_kind)
        grouping = 'reference+episode'
    report = {'seed': seed, 'grouping': grouping, 'input_kind': input_kind,
              'manual_grouping': manual_grouping if any(s.get('format') == MANUAL_FORMAT for s in samples) else None,
              'missing_training_classes': sorted(set(CLASS_NAMES) - {s['label'] for s in split['train']}),
              'target_fractions': {'train': 0.8, 'val': 0.2},
              'missing_validation_classes': sorted(set(CLASS_NAMES) - {s['label'] for s in split['val']}),
              'synthetic': any(s.get('synthetic', False) for s in samples), 'splits': {}}
    if reference_details is not None:
        report['reference_split'] = reference_details
    for name, rows in split.items():
        for label in CLASS_NAMES:
            (destination / name / label).mkdir(parents=True, exist_ok=True)
        copied = []
        for row in rows:
            target = destination / name / row['label'] / (row['sha256'] + '.png')
            # Re-encode PNG so the extension is accurate, and support Unicode paths on Windows.
            ok, encoded = cv2.imencode('.png', read_image(row['path']))
            if not ok:
                raise ValueError(f'无法编码 {row["path"]}')
            encoded.tofile(target)
            copied.append({**row, 'path': target})
        (destination / name / 'index.jsonl').write_text(''.join(
            json.dumps({**row, 'source_path': str(row['path']),
                        'prepared_path': (Path(name) / row['label'] / (row['sha256'] + '.png')).as_posix()},
                       ensure_ascii=False, default=str) + '\n' for row in rows
        ), encoding='utf-8')
        split[name] = copied
        counts = Counter(s['label'] for s in rows)
        report['splits'][name] = {'sessions': sorted({s['session_id'] for s in rows}),
                                  'episode_count': len({(s['session_id'], s['episode_id']) for s in rows}),
                                  'total': len(rows), 'fraction': len(rows) / len(samples),
                                  'counts': {label: counts[label] for label in CLASS_NAMES}}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'split.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return split, report
