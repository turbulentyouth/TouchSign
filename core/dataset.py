"""Lossless, labelled sensor recordings shared by collection and training.

Each session owns one verified unloaded reference. Splits must use session_id,
never individual frames from the same press. Images are BGR PNGs on disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.zodiac import ZODIAC_KEYS

DATASET_LABELS = (*ZODIAC_KEYS, 'unknown', 'no_contact')


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec='milliseconds')


def _encode_png(frame: np.ndarray) -> bytes:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('需要 uint8 BGR 三通道传感器图像')
    ok, encoded = cv2.imencode('.png', frame)
    if not ok:
        raise OSError('无法编码传感器图像')
    return encoded.tobytes()


class DatasetWriter:
    """Append-only writer: new sessions never overwrite earlier recordings."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.session_id: str | None = None
        self.session_dir: Path | None = None
        self.baseline_path: str | None = None
        self.baseline_sha256: str | None = None
        self._shape: tuple[int, ...] | None = None
        self._baseline: np.ndarray | None = None
        self.counts: Counter[str] = Counter()
        self._lock = threading.RLock()

    def start_session(
        self, baseline: np.ndarray, *, camera: dict[str, Any] | None = None,
        notes: str = '', contact_settings: dict[str, Any] | None = None,
    ) -> str:
        encoded = _encode_png(baseline)
        with self._lock:
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:10]
            folder = self.root / session_id
            folder.mkdir(parents=True, exist_ok=False)
            baseline_file = folder / 'baseline.png'
            baseline_file.write_bytes(encoded)
            baseline_hash = hashlib.sha256(encoded).hexdigest()
            metadata = {
                'schema_version': 1, 'session_id': session_id, 'created_at': _timestamp(),
                'camera': camera or {}, 'notes': notes,
                'baseline': baseline_file.relative_to(self.root).as_posix(),
                'baseline_sha256': baseline_hash, 'frame_shape': list(baseline.shape),
                'input_kind': 'absdiff_bgr_v1',
                'contact_settings': contact_settings or {},
                'labels': list(DATASET_LABELS),
            }
            (folder / 'session.json').write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            self.session_id = session_id
            self.session_dir = folder
            self.baseline_path = metadata['baseline']
            self.baseline_sha256 = baseline_hash
            self._shape = baseline.shape
            self._baseline = baseline.copy()
            self.counts.clear()
            return session_id

    def save(
        self, frame: np.ndarray, *, label: str, episode_id: str,
        contact_score: float, contact_area: float, frame_sequence: int = 0,
        contact: bool | None = None, stable: bool = False,
    ) -> dict[str, Any]:
        if label not in DATASET_LABELS:
            raise ValueError(f'不支持的数据标签：{label!r}')
        if not episode_id or not isinstance(episode_id, str):
            raise ValueError('episode_id 不能为空')
        if not np.isfinite(contact_score) or not np.isfinite(contact_area):
            raise ValueError('接触测量必须为有限数值')
        # This is a second guard independent of GUI button enable/disable state.
        if not isinstance(contact, bool):
            raise ValueError('必须提供经过接触检测的布尔 contact 状态')
        if (label == 'no_contact') == contact:
            raise ValueError('当前接触状态与所选标签不匹配')
        with self._lock:
            if self.session_dir is None or self.session_id is None:
                raise RuntimeError('必须先完成空载校准并建立采集会话')
            if frame.shape != self._shape:
                raise ValueError('图像分辨率已改变，请重新校准')
            from core.contact import make_difference

            encoded = _encode_png(make_difference(frame, self._baseline))
            raw_encoded = _encode_png(frame)
            folder = self.session_dir / 'difference' / label
            raw_folder = self.session_dir / 'raw' / label
            folder.mkdir(parents=True, exist_ok=True)
            raw_folder.mkdir(parents=True, exist_ok=True)
            file_name = uuid.uuid4().hex + '.png'
            image_file = folder / file_name
            raw_file = raw_folder / file_name
            row = {
                'schema_version': 1, 'image': image_file.relative_to(self.root).as_posix(),
                'raw_image': raw_file.relative_to(self.root).as_posix(),
                'input_kind': 'absdiff_bgr_v1',
                'label': label, 'session_id': self.session_id, 'episode_id': episode_id,
                'timestamp': _timestamp(), 'contact_score': float(contact_score),
                'contact_area': float(contact_area), 'contact': contact,
                'contact_detected': contact, 'stable': bool(stable),
                'frame_sequence': int(frame_sequence), 'baseline': self.baseline_path,
                'baseline_sha256': self.baseline_sha256,
                'image_sha256': hashlib.sha256(encoded).hexdigest(),
            }
            try:
                image_file.write_bytes(encoded)
                raw_file.write_bytes(raw_encoded)
                with (self.session_dir / 'manifest.jsonl').open('a', encoding='utf-8', newline='\n') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                image_file.unlink(missing_ok=True)
                raw_file.unlink(missing_ok=True)
                raise
            self.counts[label] += 1
            return row


def dataset_summary(root: str | Path) -> dict[str, Any]:
    """Read lightweight metadata only; count sessions per class for split readiness."""
    counts: Counter[str] = Counter()
    sessions: dict[str, set[str]] = {label: set() for label in DATASET_LABELS}
    invalid_lines = 0
    for manifest in Path(root).glob('*/manifest.jsonl'):
        for line in manifest.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                label, session_id = row['label'], row['session_id']
                if label not in DATASET_LABELS or not isinstance(session_id, str):
                    raise ValueError('invalid label/session')
                counts[label] += 1
                sessions[label].add(session_id)
            except (KeyError, ValueError, TypeError):
                invalid_lines += 1
    return {
        'counts': dict(counts), 'sessions_per_label': {label: len(values) for label, values in sessions.items()},
        'invalid_lines': invalid_lines,
    }
