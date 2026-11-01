"""OpenCV 摄像头读帧。原生采集直接使用 BGR，网页识别界面按需编码 JPEG。"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_SCAN_MAX_INDEX = 6
DEFAULT_JPEG_QUALITY = 82
DEFAULT_CAPTURE_FPS = 20.0


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """一台被扫描到的摄像头。"""

    index: int
    name: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    backend: str = ''
    available: bool = True

    @property
    def display_name(self) -> str:
        return self.name.strip() or f'摄像头 #{self.index}'

    @property
    def resolution(self) -> str:
        if self.width > 0 and self.height > 0:
            return f'{self.width} × {self.height}'
        return '分辨率未知'

    @property
    def short_label(self) -> str:
        """下拉框使用的紧凑标签，例如 ``#0 · Integrated Camera``。"""
        name = self.display_name
        if len(name) > 18:
            name = name[:17].rstrip() + '…'
        label = f'#{self.index} · {name}' if self.name else name
        if not self.available:
            label += '（不可用）'
        return label

    @property
    def full_label(self) -> str:
        label = f'{self.display_name} · {self.resolution}'
        if self.name:
            label = f'#{self.index} · {label}'
        if self.fps > 0:
            label += f' · {self.fps:.0f} FPS'
        if not self.available:
            label += ' · 不可用'
        return label


def _backends() -> tuple[tuple[int, str], ...]:
    if sys.platform == 'win32':
        return ((cv2.CAP_DSHOW, 'DSHOW'), (cv2.CAP_MSMF, 'MSMF'))
    return ((cv2.CAP_ANY, 'AUTO'),)


def _try_capture(index: int, backend: int) -> cv2.VideoCapture | None:
    try:
        capture = cv2.VideoCapture(index, backend)
    except cv2.error:
        return None
    if not capture.isOpened():
        capture.release()
        return None
    # 保持原训练数据与采集界面使用的传感器分辨率。
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    return capture


def _open_capture(camera: CameraInfo) -> cv2.VideoCapture | None:
    """按 DirectShow → Media Foundation 的顺序打开指定序号的摄像头。"""
    backends = list(_backends())
    if camera.backend:
        backends.sort(key=lambda item: item[1] != camera.backend)

    for backend, _label in backends:
        capture = _try_capture(camera.index, backend)
        if capture is not None:
            return capture
    return None


def _grab_frame(capture: cv2.VideoCapture, attempts: int = 6) -> np.ndarray | None:
    for attempt in range(attempts):
        try:
            ok, frame = capture.read()
        except cv2.error:
            return None
        if ok and frame is not None and frame.size:
            return frame
        if attempt + 1 < attempts:
            time.sleep(0.08)
    return None


def _probe_camera(index: int) -> CameraInfo | None:
    for backend, label in _backends():
        capture = _try_capture(index, backend)
        if capture is None:
            continue
        try:
            frame = _grab_frame(capture)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
            except cv2.error:
                fps = 0.0
            if not (0.0 < fps < 240.0):
                fps = 0.0
            return CameraInfo(
                index=index,
                name='',
                width=int(width),
                height=int(height),
                fps=round(fps, 1),
                backend=label,
            )
        finally:
            capture.release()
    return None


def scan_cameras(*, max_index: int = DEFAULT_SCAN_MAX_INDEX) -> list[CameraInfo]:
    """按设备编号探测可读的摄像头，供主界面的设备列表使用。"""
    cameras = []
    for index in range(max(1, int(max_index))):
        info = _probe_camera(index)
        if info is not None:
            cameras.append(info)
    return cameras


@dataclass(frozen=True, slots=True)
class FramePacket:
    serial: int
    sequence: int
    timestamp: float
    frame: np.ndarray


class CameraStream:
    """在线程中打开、读取和释放单台摄像头。"""

    def __init__(
        self,
        camera: CameraInfo,
        *,
        quality: int = DEFAULT_JPEG_QUALITY,
        target_fps: float = DEFAULT_CAPTURE_FPS,
        serial: int = 1,
        encode_jpeg: bool = True,
    ) -> None:
        self.camera = camera
        self.serial = serial
        self._encode_jpeg = encode_jpeg
        self._quality = max(30, min(95, int(quality)))
        self._interval = 1.0 / max(1.0, float(target_fps))
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._stamp: float | None = None
        self._stamps: deque[float] = deque(maxlen=90)
        self._error: str | None = None
        self._opened = False

    # ------------------------------------------------------------------ 生命周期
    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f'touchsign-camera-{self.camera.index}',
            daemon=True,
        )
        self._thread.start()

    def wait_ready(self, timeout: float = 4.0) -> bool:
        """等待采集线程完成设备打开，返回是否成功。"""
        self._ready.wait(timeout)
        return self._opened

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.5)
        with self._lock:
            self._jpeg = None
            self._frame = None
            self._stamp = None
            self._sequence = 0
            self._stamps.clear()

    # ------------------------------------------------------------------ 数据访问
    def snapshot(self) -> tuple[int, int, bytes | None]:
        """返回 ``(流序号, 帧序号, JPEG)``，用于 MJPEG 推送去重。"""
        with self._lock:
            return self.serial, self._sequence, self._jpeg

    def latest_frame(self) -> np.ndarray | None:
        """最新一帧 BGR 图像（只读，供后续接触检测/推理使用）。"""
        packet = self.frame_packet()
        return None if packet is None else packet.frame

    def frame_packet(self) -> FramePacket | None:
        """Atomically read pixels, sequence and monotonic acquisition timestamp."""
        with self._lock:
            if self._frame is None or self._stamp is None:
                return None
            return FramePacket(self.serial, self._sequence, self._stamp, self._frame.copy())

    def statistics(self) -> tuple[float, float | None]:
        """返回 ``(实测帧率, 距最后一帧的秒数)``。"""
        now = time.monotonic()
        with self._lock:
            age = None if self._stamp is None else now - self._stamp
            stamps = [stamp for stamp in self._stamps if now - stamp <= 2.0]
        if len(stamps) >= 2:
            span = stamps[-1] - stamps[0]
            fps = (len(stamps) - 1) / span if span > 0 else 0.0
        else:
            fps = 0.0
        return fps, age

    # ------------------------------------------------------------------ 采集线程
    def _run(self) -> None:
        capture = _open_capture(self.camera)
        if capture is None:
            self._error = f'无法打开摄像头 {self.camera.display_name}'
            self._ready.set()
            return
        self._capture = capture
        self._opened = True
        self._ready.set()

        failures = 0
        try:
            while not self._stop.is_set():
                capture = self._capture
                if capture is None:
                    # 上一次重连失败，稍后再试。
                    if self._stop.wait(1.0):
                        break
                    self._capture = _open_capture(self.camera)
                    self._error = None if self._capture is not None else self._error
                    continue

                started = time.monotonic()
                try:
                    ok, frame = capture.read()
                except cv2.error:
                    ok, frame = False, None

                if not ok or frame is None or not frame.size:
                    failures += 1
                    if failures >= 15:
                        LOGGER.warning('摄像头 %s 连续读取失败，尝试重连', self.camera.display_name)
                        failures = 0
                        try:
                            capture.release()
                        except Exception:  # noqa: BLE001 - 释放失败不应中断线程
                            pass
                        self._capture = None
                        self._error = f'摄像头 {self.camera.display_name} 信号中断'
                        # A reconnect changes acquisition conditions: invalidate calibration.
                        with self._lock:
                            self.serial += 1_000_000
                            self._frame = None
                            self._jpeg = None
                            self._stamp = None
                    else:
                        time.sleep(0.03)
                    continue

                failures = 0
                self._error = None
                self._publish(frame)

                remaining = self._interval - (time.monotonic() - started)
                if remaining > 0 and self._stop.wait(remaining):
                    break
        finally:
            self._release_capture()
            self._opened = False

    def _publish(self, frame: np.ndarray) -> None:
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        jpeg = None
        if self._encode_jpeg:
            try:
                encoded, buffer = cv2.imencode(
                    '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality],
                )
            except cv2.error:
                return
            if not encoded:
                return
            jpeg = buffer.tobytes()
        now = time.monotonic()
        with self._lock:
            self._frame = frame
            self._jpeg = jpeg
            self._sequence += 1
            self._stamp = now
            self._stamps.append(now)

    def _release_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001
                pass


class CameraManager:
    """全局摄像头管理器：保证同一时刻只打开一路采集。"""

    def __init__(self, *, serial: int = 0) -> None:
        self._control_lock = threading.RLock()
        self._stream: CameraStream | None = None
        self._cameras: list[CameraInfo] = []
        self._serial = serial
        self._owner: object | None = None

    @property
    def cameras(self) -> list[CameraInfo]:
        return list(self._cameras)

    @property
    def stream(self) -> CameraStream | None:
        return self._stream

    def scan(self, *, max_index: int = DEFAULT_SCAN_MAX_INDEX) -> list[CameraInfo]:
        """重新扫描摄像头；扫描前会释放当前采集以允许设备被重新打开。"""
        with self._control_lock:
            self.close()
            self._cameras = scan_cameras(max_index=max_index)
            return list(self._cameras)

    def select(
        self,
        camera: CameraInfo,
        *,
        quality: int = DEFAULT_JPEG_QUALITY,
        target_fps: float = DEFAULT_CAPTURE_FPS,
        owner: object | None = None,
        encode_jpeg: bool = True,
    ) -> CameraStream:
        """打开指定摄像头并返回采集流（不会等待第一帧）。

        ``owner`` 用于页面关闭时判断是否由自己持有采集流，避免多标签页之间
        互相停止对方的摄像头。
        """
        with self._control_lock:
            self.close()
            self._serial += 1
            stream = CameraStream(
                camera,
                quality=quality,
                target_fps=target_fps,
                serial=self._serial,
                encode_jpeg=encode_jpeg,
            )
            self._stream = stream
            self._owner = owner
            stream.start()
            return stream

    def close(self) -> None:
        with self._control_lock:
            stream = self._stream
            self._stream = None
            self._owner = None
        if stream is not None:
            stream.stop()

    def close_if_owner(self, owner: object) -> None:
        """仅在采集流属于 ``owner`` 时关闭，页面卸载时调用。"""
        with self._control_lock:
            if self._owner is not owner:
                return
        self.close()

    def snapshot(self) -> tuple[int, int, bytes | None]:
        stream = self._stream
        if stream is None:
            return 0, 0, None
        return stream.snapshot()

    def statistics(self) -> tuple[float, float | None] | None:
        stream = self._stream
        return None if stream is None else stream.statistics()

    def latest_frame(self) -> np.ndarray | None:
        stream = self._stream
        return None if stream is None else stream.latest_frame()

    def frame_packet(self) -> FramePacket | None:
        stream = self._stream
        return None if stream is None else stream.frame_packet()


camera_manager = CameraManager()


def describe_cameras(cameras: list[CameraInfo]) -> str:
    """调试输出：把扫描结果格式化为多行文本。"""
    if not cameras:
        return '未检测到可用摄像头。'
    lines: list[str] = []
    for camera in cameras:
        state = '可用' if camera.available else '不可用'
        lines.append(
            f'  #{camera.index}  {camera.display_name}  '
            f'{camera.resolution}  {camera.fps:.0f} FPS  [{state}]'
        )
    return '\n'.join(lines)
