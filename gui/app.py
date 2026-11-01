"""Real sensor → calibrated contact gate → ONNX inference → live NiceGUI view."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from fastapi.responses import Response, StreamingResponse
from nicegui import app, run, ui

from core.camera import CameraInfo, camera_manager, describe_cameras
from core.config import Config
from core.contact import make_difference
from core.inference import RAW_INPUT_KIND
from core.state_machine import RecognitionEngine
from core.zodiac import ZODIAC_BY_KEY
from gui.components import AppFooter, ResultPanel, SensorPanel, TopBar, ZodiacMatrix
from gui.dom import Text, element
from gui.theme import apply_theme
from gui.view_models import Candidate, ContactReading, Prediction, RecognitionRecord

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / 'assets'
PLACEHOLDER_URL = '/assets/blank-sensor.svg'
_MJPEG_PART = b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n'


async def _mjpeg_stream():
    last_token = (0, 0)
    while True:
        serial, sequence, jpeg = camera_manager.snapshot()
        if jpeg is not None and (serial, sequence) != last_token:
            last_token = serial, sequence
            yield _MJPEG_PART % len(jpeg) + jpeg + b'\r\n'
        else:
            await asyncio.sleep(0.03)


class TouchSignView:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.engine = RecognitionEngine(config)
        self.connected = True
        self.scanning = False
        self.calibrating = False
        self.analyzing = False
        self.loading_model = False
        self.streaming = False
        self.signal_lost = False
        self.camera: CameraInfo | None = None
        self.cameras = camera_manager.cameras
        self.records: list[RecognitionRecord] = []
        self.current: Prediction | None = None
        self.contact = ContactReading()
        self._waveform: deque[float] = deque([0.0] * 30, maxlen=30)
        self._stream_started = 0.0
        self._stream_serial = 0
        self._generation = 0
        self._pending_reset = True
        self._last_token = (0, 0)
        self.difference_preview = self.engine.model is None or self.engine.model.input_kind != RAW_INPUT_KIND
        self.stream_url = ''

        self.app = element('div', 'app runtime-app')
        with self.app:
            self.topbar = TopBar(on_refresh=self._refresh_cameras,
                                 on_connect=self._toggle_connection,
                                 on_camera_change=self._change_camera)
            self._build_runtime_controls()
            with element('main', 'workspace'):
                self.sensor = SensorPanel(config.sensor, PLACEHOLDER_URL, self.contact)
                self.result = ResultPanel(config.recognition.confidence_threshold,
                                          None, self.records, 0.0)
            self.matrix = ZodiacMatrix('')
            self.footer = AppFooter()

        cached = next((item for item in self.cameras if item.available), None)
        self.topbar.set_cameras(self.cameras, cached.index if cached else None)
        self.topbar.set_clock(datetime.now())
        self._update_model_status()
        self._clear_result('请移开物体，完成空载校准后开始识别', '待校准')
        self.topbar.set_status('请先进行空载校准', 'CALIBRATION REQUIRED')
        ui.timer(1.0, lambda: self.topbar.set_clock(datetime.now()))
        ui.timer(0.1, self._tick)
        ui.context.client.on_disconnect(self._on_disconnect)
        if cached is not None:
            self._activate_camera(cached)
        else:
            self.sensor.show_placeholder('未检测到可用传感器，请连接设备后刷新')

    def _build_runtime_controls(self) -> None:
        with element('div', 'runtime-controls'):
            self.calibration_button = element('button', 'runtime-button').props(
                'type="button" title="先移开所有物体，再采集稳定空载基线"')
            with self.calibration_button:
                self.calibration_text = Text('span', '空载校准')
            self.calibration_button.on('click', self._calibrate)
            self.preview_button = element('button', 'runtime-button').props('type="button"')
            with self.preview_button:
                self.preview_text = Text('span', self._preview_label)
            self.preview_button.on('click', self._toggle_preview)
            self.reload_button = element('button', 'runtime-button').props('type="button"')
            with self.reload_button:
                Text('span', '重新加载模型')
            self.reload_button.on('click', lambda: self._reload_model())
            button = element('button', 'runtime-button').props('type="button"')
            with button:
                Text('span', '模型路径')
            button.on('click', self._show_model_settings)
            self.model_status = Text('span', '', 'runtime-model-status')
            self.runtime_hint = Text('span', '校准前请移开所有物体', 'runtime-hint')

    def _update_model_status(self) -> None:
        self.model_status.text = self.engine.model_status
        self.model_status.props('title=' + json.dumps(str(self.engine.model_path), ensure_ascii=False))

    async def _tick(self) -> None:
        if self.analyzing or self.calibrating or self.scanning or self.loading_model:
            return
        self.analyzing = True
        generation = self._generation
        try:
            if self._pending_reset:
                await run.io_bound(self.engine.reset, clear_calibration=True)
                if generation != self._generation:
                    return
                self._pending_reset = False
            packet = camera_manager.frame_packet() if self.streaming and self.connected else None
            if packet is not None and packet.serial != self._stream_serial:
                # CameraStream changes its serial after automatic recovery. The new
                # frames may have a different exposure; never reuse the old baseline.
                self._stream_serial = packet.serial
                self._generation += 1
                recovery_generation = self._generation
                self._pending_reset = True
                self._last_token = (0, 0)
                self._clear_result('传感器已重新连接，请重新进行空载校准', '待校准')
                self.sensor.show_placeholder('传感器重新连接，旧基线已失效')
                await run.io_bound(self.engine.reset, clear_calibration=True)
                if recovery_generation == self._generation:
                    self._pending_reset = False
                return
            fresh = (packet is not None and time.monotonic() - packet.timestamp
                     < self.config.recognition.max_frame_age)
            if not fresh:
                if self.streaming and time.monotonic() - self._stream_started < 4.0:
                    return
                state = await run.io_bound(self.engine.process, None,
                                          sequence=0, timestamp=None, stream_id=self._stream_serial)
                if state is not None and generation == self._generation:
                    self._render_state(state)
                    self.sensor.show_placeholder('无实时信号，识别已停止')
                    self.signal_lost = True
                return
            token = packet.serial, packet.sequence
            if token == self._last_token:
                return
            self._last_token = token
            state = await run.io_bound(self.engine.process, packet.frame,
                                      sequence=packet.sequence, timestamp=packet.timestamp,
                                      stream_id=packet.serial)
            if state is None or generation != self._generation or self.calibrating:
                return
            self.signal_lost = False
            statistics = camera_manager.statistics()
            if statistics is not None:
                self.sensor.set_feed_rate(statistics[0])
            self._render_preview(packet.frame)
            self._render_state(state)
        except Exception as exc:
            LOGGER.exception('实时识别失败')
            self._clear_result(f'识别已停止：{exc}', '识别异常')
            self.topbar.set_status('识别异常', 'ERROR')
        finally:
            self.analyzing = False

    def _render_preview(self, frame) -> None:
        if not self.difference_preview:
            if not self.sensor._streaming and self.camera is not None:
                self.sensor.show_stream(self.stream_url, self.camera)
            self.sensor.set_signal(True)
            return
        baseline = self.engine.detector.baseline
        if baseline is None:
            self.sensor.show_placeholder('差分图需要空载基线 · 请先移开物体并校准')
            return
        difference = make_difference(frame, baseline)
        ok, buffer = cv2.imencode('.jpg', difference,
                                 [cv2.IMWRITE_JPEG_QUALITY, self.config.sensor.jpeg_quality])
        if ok:
            encoded = base64.b64encode(buffer).decode('ascii')
            self.sensor.image.props(f'src="data:image/jpeg;base64,{encoded}" alt="实时触觉差分图"')
            self.sensor._streaming = True
            self.sensor.view.classes(add='is-live')
            self.sensor._set_chip('DIFF', 'live')
            self.sensor._set_note('与空载基线的绝对差分')

    def _render_state(self, state) -> None:
        contact = state.contact
        self._waveform.append(float(contact.score))
        self.contact = ContactReading(
            active=contact.active, area_pct=100 * contact.area_fraction,
            score=contact.score, stability_pct=100 * contact.stability,
            duration_s=contact.duration_s, x=contact.center_x, y=contact.center_y,
            waveform=tuple(self._waveform),
        )
        self.sensor.set_reading(self.contact)
        self.runtime_hint.text = state.message
        self.topbar.set_status(state.message, state.status.upper())
        self._update_model_status()
        prediction = state.prediction
        preview = state.status == 'preview'
        if preview and prediction is not None and prediction.predicted_key in ('unknown', 'no_contact'):
            self.current = None
            self.matrix.set_active('')
            label = '未知物体' if prediction.predicted_key == 'unknown' else '空载'
            self.result.render_negative_preview(label, state.message, prediction.confidence, prediction.latency_ms)
            return
        if state.status not in ('preview', 'recognized') or prediction is None or not prediction.accepted:
            self._clear_result(state.message, {
                'no_signal': '无信号', 'uncalibrated': '待校准', 'no_contact': '无接触',
                'unstable': '接触未稳定', 'model_missing': '模型未就绪', 'unknown': '已拒识',
                'confirming': '连续确认中', 'error': '识别异常',
            }.get(state.status, '未识别'))
            return
        zodiac = ZODIAC_BY_KEY.get(prediction.key)
        if zodiac is None:
            self._clear_result('模型类别映射无效，已拒识', '已拒识')
            return
        now = datetime.now()
        current = Prediction(
            zodiac=zodiac, confidence=100 * prediction.confidence,
            candidates=tuple(Candidate(ZODIAC_BY_KEY[key], 100 * probability)
                             for key, probability in prediction.candidates if key in ZODIAC_BY_KEY),
            latency_ms=prediction.latency_ms, accepted=True,
            record_id=state.record_id, timestamp=now,
        )
        animate = self.current is None or self.current.zodiac.key != zodiac.key
        self.current = current
        self.result.render(current, self.contact.stability_pct, animate=animate, preview=preview)
        self.matrix.set_active(zodiac.key)
        if state.new_record and not preview:
            self.records.insert(0, RecognitionRecord(zodiac, current.confidence, now,
                                                      current.record_id, True))
            self.records = self.records[:self.config.recognition.history_limit]
            self.result.render_records(self.records)

    def _clear_result(self, message: str, status: str = '未识别') -> None:
        self.current = None
        self.result.render_unknown(message)
        self.result.result_state_text.text = status
        self.matrix.set_active('')

    async def _calibrate(self) -> None:
        if self.calibrating or self.scanning or self.loading_model:
            return
        if not self.streaming or not self.connected:
            ui.notify('请先连接传感器', type='warning')
            return
        self.calibrating = True
        self._generation += 1
        generation = self._generation
        self._clear_result('正在采集空载基线，请保持表面无物体', '校准中')
        self.calibration_button.props('disabled')
        self.calibration_text.text = '校准中…'
        required_frames = self.config.contact.calibration_frames
        self.runtime_hint.text = f'请保持表面无物体，正在采集 {required_frames} 帧空载图像'
        frames = []
        last_sequence = -1
        deadline = time.monotonic() + max(12.0, required_frames / 3.0)
        try:
            # Let an already-running inference/reset finish before collecting a
            # new reference, so an older worker cannot erase the new calibration.
            while self.analyzing:
                await asyncio.sleep(0.01)
            await run.io_bound(self.engine.reset, clear_calibration=True)
            while len(frames) < required_frames and time.monotonic() < deadline:
                if generation != self._generation or not self.connected:
                    raise ValueError('设备发生变化，请重新校准')
                packet = camera_manager.frame_packet()
                if packet is not None:
                    if packet.serial != self._stream_serial:
                        raise ValueError('设备重新连接，请重新校准')
                    if (packet.sequence != last_sequence and time.monotonic() - packet.timestamp
                            < self.config.recognition.max_frame_age):
                        frames.append(packet.frame)
                        last_sequence = packet.sequence
                        self.calibration_text.text = f'校准 {len(frames)}/{required_frames}'
                await asyncio.sleep(0.04)
            if len(frames) < required_frames:
                raise ValueError('实时帧不足，请检查传感器连接')
            await run.io_bound(self.engine.calibrate, frames)
            latest = camera_manager.frame_packet()
            if (generation != self._generation or latest is None
                    or latest.serial != self._stream_serial
                    or time.monotonic() - latest.timestamp >= self.config.recognition.max_frame_age):
                await run.io_bound(self.engine.reset, clear_calibration=True)
                self._pending_reset = True
                raise ValueError('校准结束时信号已变化，请重新校准')
            self._pending_reset = False
            self._last_token = (0, 0)
            self._waveform.clear()
            self._waveform.extend([0.0] * 30)
            ui.notify('空载校准完成，可以接触待识别牌子', type='positive')
        except Exception as exc:
            self._pending_reset = True
            self._clear_result(f'校准失败：{exc}', '校准失败')
            ui.notify(f'校准失败：{exc}', type='negative')
        finally:
            self.calibrating = False
            self.calibration_button.props(remove='disabled')
            self.calibration_text.text = '重新空载校准' if self.engine.detector.baseline is not None else '空载校准'

    @property
    def _preview_label(self) -> str:
        return '差分图 · 切换原图' if self.difference_preview else '原始图 · 切换差分'

    def _toggle_preview(self) -> None:
        self.difference_preview = not self.difference_preview
        self.preview_text.text = self._preview_label
        self._last_token = (0, 0)
        if not self.difference_preview and self.streaming and self.camera is not None:
            self.sensor.show_stream(self.stream_url, self.camera, pending=True)
        elif self.difference_preview:
            self.sensor.show_placeholder('等待差分图')

    def _show_model_settings(self) -> None:
        with ui.dialog() as dialog, ui.card().classes('w-[650px] max-w-full'):
            ui.label('ONNX 模型路径').classes('text-lg')
            ui.label('选择训练导出的 ONNX 文件，同目录需有同名 JSON 文件。').classes('text-sm')
            path_input = ui.input('模型文件', value=str(self.engine.model_path)).classes('w-full')
            preview = ui.checkbox('预览模式（跳过拒识）', value=self.engine.preview)
            ui.label('预览时，稳定接触后显示当前帧得分最高的类别。').classes('text-sm')
            with ui.row():
                async def apply_path():
                    if await self._reload_model(path_input.value, preview=preview.value):
                        dialog.close()
                ui.button('加载模型', on_click=apply_path)
                ui.button('关闭', on_click=dialog.close).props('flat')
        dialog.open()

    async def _reload_model(self, path: str | None = None, *, preview: bool | None = None) -> bool:
        if self.calibrating or self.loading_model or self.scanning:
            ui.notify('请等待当前操作完成', type='warning')
            return False
        self.loading_model = True
        self._generation += 1
        self._clear_result('正在加载 ONNX 模型', '模型加载中')
        self.reload_button.props('disabled')
        try:
            while self.analyzing:
                await asyncio.sleep(0.01)
            await run.io_bound(self.engine.reload_model, path, preview=preview)
            self._last_token = (0, 0)
            self._update_model_status()
            ready = self.engine.model is not None
            ui.notify(self.engine.model_status, type='positive' if ready else 'negative')
            return ready
        except Exception as exc:
            self._clear_result(f'模型加载失败：{exc}', '模型未就绪')
            ui.notify(f'模型加载失败：{exc}', type='negative')
            return False
        finally:
            self.loading_model = False
            self.reload_button.props(remove='disabled')

    async def _refresh_cameras(self) -> None:
        if self.scanning:
            return
        self.scanning = True
        previous = self.camera.index if self.camera else None
        self._stop_stream()
        self.topbar.set_scanning(True)
        self.topbar.set_searching()
        self.sensor.show_placeholder('正在搜索传感器')
        try:
            self.cameras = await run.io_bound(camera_manager.scan,
                                             max_index=self.config.sensor.scan_max_index) or []
            available = [camera for camera in self.cameras if camera.available]
            selected = next((camera for camera in available if camera.index == previous), None)
            selected = selected or (available[0] if available else None)
            self.topbar.set_cameras(self.cameras, selected.index if selected else None)
            if selected:
                self._activate_camera(selected)
            else:
                self.camera = None
                self.sensor.set_camera(None)
                self.sensor.show_placeholder('未检测到可用传感器')
        except Exception as exc:
            LOGGER.exception('摄像头扫描失败')
            self.sensor.show_placeholder(f'搜索失败：{exc}')
        finally:
            self.scanning = False
            self.topbar.set_scanning(False)

    def _activate_camera(self, camera: CameraInfo) -> None:
        self._stop_stream()
        self.camera = camera
        self.sensor.set_camera(camera)
        self.topbar.set_selected_camera(camera.index)
        if not self.connected:
            return
        stream = camera_manager.select(camera, quality=self.config.sensor.jpeg_quality,
                                       target_fps=self.config.sensor.capture_fps, owner=self)
        self._stream_serial = stream.serial
        self.streaming = True
        self._stream_started = time.monotonic()
        self.stream_url = f'/camera/stream?v={time.time_ns()}'
        self.sensor.show_placeholder('传感器连接中 · 请移开物体后空载校准')
        if not self.difference_preview:
            self.sensor.show_stream(self.stream_url, camera, pending=True)

    def _change_camera(self, event) -> None:
        value = event.args[0] if isinstance(event.args, list) and event.args else event.args
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        camera = next((item for item in self.cameras if item.index == index and item.available), None)
        if camera is not None:
            self._activate_camera(camera)

    def _stop_stream(self) -> None:
        self._generation += 1
        self._pending_reset = True
        camera_manager.close_if_owner(self)
        self.streaming = False
        self._last_token = (0, 0)
        self.contact = ContactReading()
        self.sensor.set_reading(self.contact)
        self._clear_result('设备已变化，请重新进行空载校准', '待校准')

    def _toggle_connection(self) -> None:
        self.connected = not self.connected
        self.topbar.set_connected(self.connected)
        self.sensor.set_connected(self.connected)
        self.app.classes(add=None if self.connected else 'is-sensor-offline',
                         remove='is-sensor-offline' if self.connected else None)
        if self.connected and self.camera is not None:
            self._activate_camera(self.camera)
        elif not self.connected:
            self._stop_stream()
            self.sensor.show_placeholder('传感器已断开，识别已停止')
            self.topbar.set_status('传感器离线', 'OFFLINE')

    def _on_disconnect(self) -> None:
        self._generation += 1
        camera_manager.close_if_owner(self)


def register_pages(config: Config) -> None:
    app.add_static_files('/assets', str(ASSETS_DIR))
    try:
        camera_manager.scan(max_index=config.sensor.scan_max_index)
        LOGGER.info('摄像头扫描：\n%s', describe_cameras(camera_manager.cameras))
    except Exception:
        LOGGER.exception('启动时摄像头扫描失败')
    app.on_shutdown(camera_manager.close)

    @app.get('/camera/stream')
    async def camera_stream() -> StreamingResponse:
        return StreamingResponse(_mjpeg_stream(),
                                 media_type='multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    @app.get('/camera/snapshot.jpg')
    async def camera_snapshot() -> Response:
        _, _, jpeg = camera_manager.snapshot()
        if jpeg is None:
            return Response(status_code=503)
        return Response(jpeg, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @ui.page('/', title=config.app.title,
             favicon=str(config.favicon_path) if config.favicon_path and config.favicon_path.is_file() else None,
             dark=True)
    def index() -> None:
        apply_theme()
        TouchSignView(config)


def run_app(config: Config, *, host: str | None = None, port: int | None = None,
            show: bool | None = None, reload: bool | None = None, native: bool | None = None) -> None:
    register_pages(config)
    favicon = config.favicon_path
    use_native = config.app.native if native is None else native
    # Native windows use an available local port unless the caller supplies one.
    run_port = port if port is not None else (None if use_native else config.app.port)
    ui.run(host=host or config.app.host, port=run_port, title=config.app.title,
           favicon=str(favicon) if favicon and favicon.is_file() else None, dark=True,
           show=config.app.show if show is None else show,
           reload=config.app.reload if reload is None else reload, native=use_native,
           window_size=(config.app.window_width, config.app.window_height) if use_native else None,
           reconnect_timeout=10.0)
