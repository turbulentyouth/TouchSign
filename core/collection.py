"""Collect calibrated sensor samples and their lossless BGR difference images."""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from core.camera import CameraManager, FramePacket
from core.contact import ContactDetector, ContactResult, make_difference
from core.dataset import DATASET_LABELS, DatasetWriter


class CollectionController:
    def __init__(self, camera: CameraManager, root: Path, *, contact_config=None) -> None:
        self.camera = camera
        self.detector = ContactDetector(contact_config)
        self.writer = DatasetWriter(root)
        self.contact_config = contact_config
        self.lock = threading.RLock()
        self.recording = False
        self.calibrating = False
        self.label: str | None = None
        self.message = '选择传感器，移开所有物体，再确认空载并校准。'
        self.contact: ContactResult | None = None
        self.saved_run = 0
        self.target = 100
        self.interval = 0.5
        self.max_per_episode = 20
        self._last_token: tuple[int, int] | None = None
        self._sample: tuple[FramePacket, np.ndarray] | None = None
        self._saved_token: tuple[int, int] | None = None
        self._last_saved_at = float('-inf')
        self._last_thumbnail: np.ndarray | None = None
        self._last_image_hash: bytes | None = None
        self._episode_id = uuid.uuid4().hex
        self._episode_count = 0
        self._episode_label: str | None = None
        self._was_active = False
        self._inactive_since = time.monotonic()
        self._raw_frame: np.ndarray | None = None
        self._selected_serial: int | None = None

    def invalidate(self, message: str) -> None:
        with self.lock:
            self.recording = False
            self.label = None
            self.detector.reset()
            self.contact = None
            self._last_token = None
            self._sample = None
            self._raw_frame = None
            self._selected_serial = None
            self.message = message

    def choose_label(self, label: str | None) -> None:
        with self.lock:
            self.recording = False
            self.label = label if label in DATASET_LABELS else None
            self.saved_run = 0
            self.message = '核对当前类别与实物，按空格保存一张。'

    def start(self, *, confirmed: bool, target: int, interval: float, max_per_episode: int) -> None:
        with self.lock:
            if self.calibrating or self.detector.baseline is None:
                raise ValueError('请先完成空载校准')
            if self.label not in DATASET_LABELS or not confirmed:
                raise ValueError('请选择类别，并确认当前实物与标签一致')
            stream = self.camera.stream
            age = None if stream is None else stream.statistics()[1]
            if age is None or not -0.1 <= age <= 0.5:
                raise ValueError('传感器没有新鲜图像，不能开始采集')
            if target < 1 or interval < 0.1 or max_per_episode < 1:
                raise ValueError('目标张数、间隔和每次接触上限无效')
            self.target, self.interval, self.max_per_episode = target, interval, max_per_episode
            self.saved_run = 0
            self.recording = True
            self.message = '采集中；只有接触状态符合标签的图像会保存。'

    def stop(self) -> None:
        with self.lock:
            self.recording = False
            self.message = f'采集已停止，本轮保存 {self.saved_run} 张。'

    def calibrate(self, *, confirmed_unloaded: bool, notes: str = '', cancel: threading.Event | None = None) -> str:
        """Collect distinct live frames; callers run this blocking method in a worker."""
        if not confirmed_unloaded:
            raise ValueError('必须确认传感器表面完全空载')
        with self.lock:
            if self.calibrating:
                raise ValueError('正在校准，请稍候')
            self.calibrating = True
            self.recording = False
            self.label = None
            self.detector.reset()
            self._sample = None
            self.contact = None
            frame_count = self.detector.config.calibration_frames
            self.message = f'正在采集 {frame_count} 帧空载参考，请勿触碰传感器…'
            stream = self.camera.stream
        try:
            if stream is None:
                raise ValueError('请先连接传感器')
            calibration_serial = stream.serial
            frames: list[np.ndarray] = []
            deadline = time.monotonic() + max(12.0, frame_count / 3.0)
            last_sequence = -1
            while len(frames) < frame_count and time.monotonic() < deadline:
                if cancel is not None and cancel.is_set():
                    raise ValueError('校准已取消')
                if self.camera.stream is not stream or stream.serial != calibration_serial:
                    raise ValueError('校准过程中传感器发生切换，请重试')
                packet = stream.frame_packet()
                if packet is not None and packet.sequence != last_sequence and time.monotonic() - packet.timestamp < 0.5:
                    frames.append(packet.frame)
                    last_sequence = packet.sequence
                time.sleep(0.01)
            if len(frames) < frame_count:
                raise ValueError('未收到足够的新鲜图像，请检查连接后重新校准')
            with self.lock:
                if self.camera.stream is not stream or stream.serial != calibration_serial:
                    raise ValueError('校准结束时传感器发生重连，请重新校准')
                baseline = self.detector.calibrate(frames)
                if cancel is not None and cancel.is_set():
                    raise ValueError('校准已取消')
                session_id = self.writer.start_session(
                    baseline, camera=asdict(stream.camera), notes=notes,
                    contact_settings=asdict(self.contact_config) if self.contact_config is not None else {},
                )
                self._selected_serial = stream.serial
                self._last_token = None
                self._saved_token = None
                self._last_saved_at = float('-inf')
                self._episode_id = uuid.uuid4().hex
                self._episode_count = 0
                self._episode_label = None
                self._was_active = False
                self._inactive_since = time.monotonic()
                self._last_thumbnail = None
                self._last_image_hash = None
                self.contact = None
                self.saved_run = 0
                self.message = '空载校准完成。请选择类别并核对标签；本次已建立新的独立会话。'
                return session_id
        except Exception as exc:
            with self.lock:
                self.detector.reset()
                self.message = f'校准失败：{exc}'
            raise
        finally:
            with self.lock:
                self.calibrating = False

    def poll(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the latest raw frame and its paired, uncompressed difference for HighGUI."""
        with self.lock:
            self._process_latest(manual=False)
            difference = self._sample[1] if self._sample is not None else None
            return self._raw_frame, difference

    def capture_once(self, *, expected_label: str | None = None) -> bool:
        """Save this keypress's current eligible frame or decline immediately."""
        with self.lock:
            if self.recording:
                return self._decline('连拍中，请先停止连拍再单拍。')
            if expected_label is not None and expected_label != self.label:
                return self._decline('类别已切换，请核对当前标签后重新拍摄。')
            if self.label not in DATASET_LABELS:
                return self._decline('请先选择与实物一致的类别。')
            return self._process_latest(manual=True)

    def _decline(self, message: str) -> bool:
        self.message = message
        return False

    def _process_latest(self, *, manual: bool) -> bool:
        self._raw_frame = None
        if self.calibrating:
            if manual:
                self.message = '正在校准，请稍候。'
            return False
        stream = self.camera.stream
        if stream is None:
            if manual or self.recording or self.detector.baseline is not None:
                self.invalidate('设备已断开，请重新连接并校准。')
            return False
        try:
            packet = stream.frame_packet()
            if packet is None or not -0.1 <= time.monotonic() - packet.timestamp <= 0.5:
                if manual or self.recording or self.detector.baseline is not None:
                    self.invalidate('没有新鲜传感器图像，请检查连接并重新校准。')
                return False
            self._raw_frame = packet.frame
            if self.detector.baseline is None:
                if manual:
                    self.message = '请先移开物体并完成空载校准。'
                return False
            if packet.serial != self._selected_serial or self.camera.stream is not stream:
                self.invalidate('传感器已切换，请重新进行空载校准。')
                return False
            token = (packet.serial, packet.sequence)
            if self._last_token is not None and token < self._last_token:
                self.invalidate('传感器帧序号已重置，请重新校准。')
                return False
            if token != self._last_token:
                self._observe(packet)
            elif not manual:
                return False
            if self._sample is None:
                # _observe recorded why this frame cannot be saved.
                if manual and self.contact is not None:
                    self.message = self.contact.reason
                return False
            if not (manual or self.recording):
                return False
            if token == self._saved_token:
                return self._decline('该帧已保存，相机更新画面后可再按空格。')
            sample, difference = self._sample
            return self._save_sample(sample, difference, manual=manual)
        except Exception as exc:
            self.recording = False
            self._sample = None
            return self._decline(f'采集失败：{exc}')

    def _observe(self, packet: FramePacket) -> None:
        """Pair the contact measurement with the exact frame it describes."""
        self._last_token = (packet.serial, packet.sequence)
        self._sample = None
        result = self.detector.process(packet.frame, timestamp=packet.timestamp)
        self.contact = result
        if not result.calibrated:
            self.invalidate(result.reason or '参考图失效，请重新校准。')
            return
        if not result.valid:
            self.message = result.reason
            return
        difference = make_difference(packet.frame, self.detector.baseline)
        self._sample = packet, difference
        if result.active != self._was_active:
            self._episode_id = uuid.uuid4().hex
            self._episode_count = 0
            self._episode_label = None
            self._last_thumbnail = None
            self._last_image_hash = None
            self._was_active = result.active
            if not result.active:
                self._inactive_since = packet.timestamp

    def _save_sample(self, packet: FramePacket, difference: np.ndarray, *, manual: bool) -> bool:
        result = self.contact
        label = self.label
        if result is None or label not in DATASET_LABELS:
            return self._decline('请先选择类别并等待有效图像。')
        if label == 'no_contact':
            if (result.active or result.area_fraction >= self.detector.config.release_area
                    or packet.timestamp - self._inactive_since < 0.3):
                return self._decline('空载类需要移开所有物体并保持空载。')
        elif not (result.active and result.stable):
            return self._decline('请将当前物体稳定按在传感器上，再拍摄。')
        if self._episode_label is not None and self._episode_label != label:
            return self._decline('本次按压已按其他类别保存，请松开并更换实物后再拍。')
        if not manual:
            if self._episode_count >= self.max_per_episode:
                return self._decline('本次接触已达连拍上限，请松开重压，空载类可新建会话。')
            interval = max(2.0, self.interval) if label == 'no_contact' else self.interval
            if packet.timestamp - self._last_saved_at < interval:
                return False
        # Keep the last saved image as the reference when switching to automatic capture.
        thumbnail = None
        image_hash = None
        if label == 'no_contact':
            image_hash = hashlib.sha256(difference).digest()
            if not manual and image_hash == self._last_image_hash:
                return self._decline('空载连拍等待画面变化。')
        else:
            thumbnail = cv2.GaussianBlur(cv2.resize(difference, (64, 64)), (5, 5), 0).astype(np.float32)
            if (not manual and self._last_thumbnail is not None
                    and np.mean(np.abs(thumbnail - self._last_thumbnail)) < 0.7):
                return self._decline('连拍画面与上一张过于相似，请改变姿态或松开重压。')
        self.writer.save(
            packet.frame, label=label, episode_id=self._episode_id,
            contact_score=result.score, contact_area=result.area_fraction,
            frame_sequence=packet.sequence, contact=result.active, stable=result.stable,
        )
        self.saved_run += 1
        self._episode_count += 1
        self._episode_label = label
        self._saved_token = packet.serial, packet.sequence
        self._last_saved_at = packet.timestamp
        self._last_thumbnail = thumbnail
        self._last_image_hash = image_hash
        if manual:
            self.message = '已保存 1 张（差分图及对应原图）。再按空格可继续拍摄。'
        else:
            self.message = f'连拍已保存 {self.saved_run}/{self.target} 张。'
            if self.saved_run >= self.target:
                self.recording = False
                self.message += ' 本轮已完成。'
        return True
