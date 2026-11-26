"""Freeze an inferred sensor frame and save a human-labelled training example."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import threading
import time
import uuid

from core.camera import FramePacket
from core.manual_collection import ManualDatasetWriter
from core.zodiac import ZODIAC_KEYS

CORRECTION_FORMAT = 'touchsign_correction_v1'


@dataclass(frozen=True)
class CorrectionSnapshot:
    packet: FramePacket
    generation: int
    captured_at: str
    details: dict


class CorrectionRecorder:
    def __init__(self, root: Path, max_frame_age: float = .5) -> None:
        self.root = root
        self.max_frame_age = max_frame_age
        self.latest: CorrectionSnapshot | None = None
        self.writer: ManualDatasetWriter | None = None
        self.count = 0
        self._identity = None
        self._timestamp: float | None = None
        self._episode: str | None = None
        self._saved: set[tuple[int, int]] = set()
        self._lock = threading.Lock()

    def observe(self, packet, state, *, generation, model, camera, threshold) -> None:
        """Packet and prediction must come from the same completed inference."""
        identity = (generation, packet.serial)
        if (identity != self._identity or self._timestamp is None
                or not 0 < packet.timestamp - self._timestamp <= self.max_frame_age):
            self._episode = None
        self._identity, self._timestamp = identity, packet.timestamp
        self.latest = None
        if not (state.contact.calibrated and state.contact.valid and state.contact.active):
            self._episode = None
            return
        if self._episode is None:
            self._episode = uuid.uuid4().hex
        # Low-confidence and confirming predictions are useful hard examples too.
        if state.prediction is None or model is None:
            return
        captured_at = datetime.fromtimestamp(
            time.time() - (time.monotonic() - packet.timestamp),
        ).astimezone().isoformat(timespec='milliseconds')
        self.latest = CorrectionSnapshot(packet, generation, captured_at, {
            'format': CORRECTION_FORMAT, 'episode_id': self._episode,
            'prediction': asdict(state.prediction),
            'recognition_status': state.status, 'recognition_message': state.message,
            'confidence_threshold': threshold, 'contact': asdict(state.contact),
            'camera': asdict(camera) if camera is not None else {},
            'model': {'path': str(model.model_path), 'manifest_path': str(model.manifest_path),
                      'preview': model.preview,
                      **{key: model.manifest.get(key) for key in
                         ('sha256', 'created_utc', 'status', 'input_kind', 'preprocess', 'image_size')}},
        })

    def freeze(self, *, generation: int, stream_serial: int) -> CorrectionSnapshot:
        snapshot = self.latest
        if (snapshot is None or snapshot.generation != generation
                or snapshot.packet.serial != stream_serial
                or not 0 <= time.monotonic() - snapshot.packet.timestamp <= self.max_frame_age):
            raise ValueError('没有新鲜的识别帧，请保持按压，出现预测后再点击保存纠错样本')
        if (snapshot.packet.serial, snapshot.packet.sequence) in self._saved:
            raise ValueError('这一帧已保存，请重新按压或等待新帧')
        return replace(snapshot, packet=replace(snapshot.packet, frame=snapshot.packet.frame.copy()),
                       details=deepcopy(snapshot.details))

    def save(self, snapshot: CorrectionSnapshot, label: str | None, notes: str = '') -> dict:
        if label not in ZODIAC_KEYS:
            raise ValueError('请选择实物的正确生肖，不能使用模型预测代替人工标签')
        with self._lock:
            token = (snapshot.packet.serial, snapshot.packet.sequence)
            if token in self._saved:
                raise ValueError('这一帧已经保存，未重复写入')
            details = {**snapshot.details, 'notes': notes.strip(),
                       'labelled_at': datetime.now().astimezone().isoformat(timespec='milliseconds'),
                       'prediction_matches_label': snapshot.details['prediction']['predicted_key'] == label}
            if self.writer is None:
                self.writer = ManualDatasetWriter(self.root, camera=details['camera'])
            row = self.writer.save(snapshot.packet, label, snapshot.captured_at, correction=details)
            self._saved.add(token)
            self.count += 1
            return {**row, 'saved_path': str(self.writer.session_dir / row['image'])}
