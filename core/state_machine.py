"""Camera contact detection, model input selection, ONNX recognition and logging."""
from __future__ import annotations
import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from core.config import Config, resolve_path
from core.contact import ContactDetector, ContactResult, make_difference
from core.inference import RAW_INPUT_KIND, ModelPrediction, OnnxClassifier
from core.rejection import TemporalConfirmation


@dataclass(frozen=True, slots=True)
class RecognitionState:
    status: str
    message: str
    contact: ContactResult = ContactResult()
    prediction: ModelPrediction | None = None
    new_record: bool = False
    record_id: str = ''


class RecognitionEngine:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._lock = threading.RLock()
        self.detector = ContactDetector(self.config.contact)
        self.confirmation = TemporalConfirmation(self.config.recognition.confirmation_frames)
        self.model_path = resolve_path(self.config.recognition.model_path)
        self.preview = self.config.recognition.preview
        self.model: OnnxClassifier | None = None
        self.model_status = '未加载模型'
        self._stream_id: int | None = None
        self._last_sequence: int | None = None
        self._last_timestamp: float | None = None
        self._record_id = ''
        self._record_key: str | None = None
        self.state = RecognitionState('uncalibrated', '请先确认空载并校准')
        self.reload_model()

    def reload_model(self, path: str | Path | None = None, *, preview: bool | None = None) -> str:
        with self._lock:
            if path:
                self.model_path = resolve_path(path)
            if preview is not None:
                self.preview = preview
            self.model = None
            self.reset(clear_calibration=False)
            try:
                configured = self.config.recognition
                manifest = None
                if configured.manifest_path and self.model_path == resolve_path(configured.model_path):
                    manifest = resolve_path(configured.manifest_path)
                self.model = OnnxClassifier(self.model_path, manifest, preview=self.preview)
                mode = '模型预览 · 未启用拒识' if self.preview else 'ONNX 就绪'
                self.model_status = f'{mode} · {self.model_path.name}'
            except Exception as exc:
                self.model_status = f'模型未就绪：{exc}'
            return self.model_status

    def reset(self, clear_calibration: bool = True) -> None:
        with self._lock:
            if clear_calibration:
                self.detector.reset()
                self._stream_id = None
            else:
                self.detector.clear_temporal()
            self.confirmation.reset()
            self._last_sequence = None
            self._last_timestamp = None
            self._record_id = ''
            self._record_key = None
            self.state = (RecognitionState('no_contact', '等待接触') if self.detector.calibrated
                          else RecognitionState('uncalibrated', '请先确认空载并校准'))

    def calibrate(self, frames: list[np.ndarray]) -> None:
        with self._lock:
            self.reset()
            self.detector.calibrate(frames)
            self.state = RecognitionState('no_contact', '空载校准完成，等待接触', ContactResult(calibrated=True, reason='未接触'))

    def _set(self, status: str, message: str, contact: ContactResult | None = None,
             prediction: ModelPrediction | None = None, new_record: bool = False) -> RecognitionState:
        self.state = RecognitionState(status, message, contact or ContactResult(), prediction, new_record,
                                      self._record_id if status == 'recognized' else '')
        return self.state

    def process(self, frame: np.ndarray | None, *, sequence: int, timestamp: float | None,
                stream_id: int = 0) -> RecognitionState:
        with self._lock:
            now = time.monotonic()
            if frame is None or timestamp is None or not np.isfinite(timestamp) or now-timestamp > self.config.recognition.max_frame_age or timestamp-now > .1:
                # Any interruption may reconnect a different sensor/exposure. Require recalibration.
                self.reset()
                return self._set('no_signal', '无实时传感器信号，请检查连接并重新空载校准')
            if self._stream_id is not None and stream_id != self._stream_id:
                self.reset()
            self._stream_id = stream_id
            if self._last_sequence is not None and sequence <= self._last_sequence:
                if sequence == self._last_sequence:
                    return replace(self.state, new_record=False)
                self.reset()
                self._stream_id = stream_id
            if self._last_timestamp is not None and (timestamp <= self._last_timestamp or timestamp-self._last_timestamp > self.config.recognition.max_frame_age):
                self.confirmation.reset()
                self.detector.clear_temporal()
            self._last_sequence, self._last_timestamp = sequence, timestamp
            contact = self.detector.process(frame, timestamp)
            if not contact.calibrated:
                self.confirmation.reset()
                return self._set('uncalibrated', contact.reason, contact)
            if not contact.valid:
                self.confirmation.reset()
                return self._set('unknown', contact.reason, contact)
            if not contact.active:
                self.confirmation.reset()
                self._record_id = ''
                self._record_key = None
                return self._set('no_contact', contact.reason, contact)
            if not contact.stable:
                self.confirmation.reset()
                return self._set('unstable', contact.reason, contact)
            if self.model is None:
                self.confirmation.reset()
                return self._set('model_missing', self.model_status, contact)
            try:
                if self.model.input_kind == RAW_INPUT_KIND:
                    model_input = frame
                else:
                    model_input = make_difference(frame, self.detector.baseline)
                prediction = self.model.predict(model_input)
            except Exception as exc:
                self.confirmation.reset()
                return self._set('error', f'推理失败：{exc}', contact)
            if self.preview:
                message = {'unknown': '模型首选未知物体', 'no_contact': '模型首选空载'}.get(
                    prediction.predicted_key, '显示当前帧首选生肖')
                return self._set('preview', message, contact, prediction)
            accepted = prediction.accepted and prediction.confidence >= self.config.recognition.confidence_threshold
            confirmed = self.confirmation.update(prediction.key, accepted)
            if not accepted:
                messages = {'unknown': '不是已知生肖牌子', 'no_contact': '模型判定无有效接触',
                            'low_confidence': '置信度不足', 'ambiguous': '生肖特征不明确',
                            'out_of_distribution': '与已采集生肖特征不匹配'}
                message = messages.get(prediction.reason, '证据不足，已拒识')
                return self._set('unknown', message, contact, prediction)
            if not confirmed:
                return self._set('confirming', f'多帧确认 {self.confirmation.count}/{self.confirmation.required}', contact)
            if self._record_key is not None and prediction.key != self._record_key:
                return self._set('unknown', '同一次接触的生肖结果发生变化，请移开牌子后重新接触', contact)
            new_record = not self._record_id
            if new_record:
                self._record_id = uuid.uuid4().hex[:12].upper()
                try:
                    self._write_record(prediction, contact)
                    self._record_key = prediction.key
                except OSError as exc:
                    self._record_id = ''
                    return self._set('error', f'识别记录保存失败：{exc}', contact)
            return self._set('recognized', '已确认生肖接触', contact, prediction, new_record)

    def _write_record(self, prediction: ModelPrediction, contact: ContactResult) -> None:
        path = resolve_path(self.config.recognition.log_path, writable=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ['timestamp', 'record_id', 'zodiac', 'confidence', 'margin', 'distance',
                  'latency_ms', 'contact_score', 'contact_area', 'model']
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open('a', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            if needs_header:
                writer.writerow(header)
            writer.writerow([datetime.now().astimezone().isoformat(), self._record_id, prediction.key,
                             prediction.confidence, prediction.margin, prediction.distance,
                             prediction.latency_ms, contact.score, contact.area_fraction, str(self.model_path)])
