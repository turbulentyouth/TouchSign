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


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or not image.size:
        raise ValueError(f'无法解码采集图像：{path}')
    return image


def read_samples(root: str | Path, *, input_kind: str = INPUT_KIND) -> list[dict]:
    """Keep the collection metadata; path and sha256 identify the selected model input."""
    if input_kind not in (INPUT_KIND, RAW_INPUT_KIND):
        raise ValueError(f'不支持的模型输入：{input_kind}')
    image_field = 'raw_image' if input_kind == RAW_INPUT_KIND else 'image'
    root = Path(root).resolve()
    manifests = sorted(root.glob('*/manifest.jsonl'))
    if not manifests:
        raise ValueError(f'没有采集数据：{root}。请先运行 collect.py，采集十二生肖、未知物体和无接触数据。')
    samples, paths, hashes = [], set(), {}
    for manifest in manifests:
        cached_baseline_path = baseline = None
        for line_no, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if input_kind == RAW_INPUT_KIND and not row.get('raw_image'):
                    raise ValueError('清单缺少原图路径 raw_image')
                image = (root / row[image_field]).resolve()
                label, session = row['label'], row['session_id']
                if not image.is_relative_to(manifest.parent.resolve()) or not image.is_file():
                    raise ValueError('图像缺失或路径越界')
                if session != manifest.parent.name or label not in CLASS_NAMES or not row.get('episode_id'):
                    raise ValueError('session_id、label 或 episode_id 无效')
                if row.get('input_kind') != INPUT_KIND:
                    raise ValueError('采集清单格式不匹配，请用新版采集界面采集')
                if row.get('contact_detected') is not (label != 'no_contact'):
                    raise ValueError('接触标注不符合类别：生肖和 unknown 必须确有接触，no_contact 必须无接触')
                if image in paths:
                    raise ValueError('清单包含重复图像路径')
                if input_kind == RAW_INPUT_KIND and row.get('image') and image == (root / row['image']).resolve():
                    raise ValueError('raw_image 与差分图指向同一文件，请核对原图记录')
                frame = read_image(image)
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
    if not eligible.any() or (not allow_train_only and not eligible.all()):
        return None
    if (totals < minimum * (1 + eligible.astype(int))).any():
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
                     + np.maximum(minimum - (totals - val), 0).sum())
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


def _grouped_split(samples: list[dict], *, seed: int, min_per_class: int):
    if min_per_class < 1:
        raise ValueError('每类的最少图像数必须为正数')
    totals = Counter(s['label'] for s in samples)
    missing = [label for label in CLASS_NAMES if totals[label] < min_per_class]
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


def prepare_dataset(raw: str | Path, destination: str | Path, *, seed: int = 42,
                    min_per_class: int = 1, input_kind: str = INPUT_KIND) -> tuple[dict, dict]:
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'输出目录非空，避免混入旧数据：{destination}')
    samples = read_samples(raw, input_kind=input_kind)
    split, grouping = _grouped_split(samples, seed=seed, min_per_class=min_per_class)
    report = {'seed': seed, 'grouping': grouping, 'input_kind': input_kind,
              'target_fractions': {'train': 0.8, 'val': 0.2},
              'missing_validation_classes': sorted(set(CLASS_NAMES) - {s['label'] for s in split['val']}),
              'synthetic': any(s.get('synthetic', False) for s in samples), 'splits': {}}
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
        split[name] = copied
        counts = Counter(s['label'] for s in rows)
        report['splits'][name] = {'sessions': sorted({s['session_id'] for s in rows}),
                                  'episode_count': len({(s['session_id'], s['episode_id']) for s in rows}),
                                  'total': len(rows), 'fraction': len(rows) / len(samples),
                                  'counts': {label: counts[label] for label in CLASS_NAMES}}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'split.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return split, report
