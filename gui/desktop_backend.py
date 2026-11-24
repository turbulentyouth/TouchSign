"""Socket-free desktop bridge. JSON commands in, immutable UI events out."""
from __future__ import annotations
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging
from pathlib import Path
import selectors
import threading
import time

from core.camera import camera_manager
from core.config import PROJECT_ROOT
from core.state_machine import RecognitionEngine
from core.zodiac import ZODIAC_BY_KEY
from robot.settings import SettingsStore, RobotSettings, joints
from robot.service import RobotService

LOGGER = logging.getLogger(__name__)


class _Selector(selectors.SelectSelector):
    def select(self, timeout=None):
        time.sleep(min(.01, max(0., timeout)) if timeout is not None else .01)
        return []


class _Loop(asyncio.SelectorEventLoop):
    """Timers/executor completions only. CAN/camera I/O is on SDK threads."""
    def __init__(self):
        super().__init__(_Selector())
    def _make_self_pipe(self): pass
    def _close_self_pipe(self): pass
    def _write_to_self(self): pass


class DesktopBackend:
    def __init__(self, config, send, settings_store=None):
        self.config, self.send = config, send
        self.store = settings_store or SettingsStore()
        self.closed = False
        self.ready = threading.Event()
        self._owner = object()
        self._loop = _Loop()
        self._vision = ThreadPoolExecutor(max_workers=1, thread_name_prefix='desktop-vision')
        self._thread = threading.Thread(target=self._run, name='desktop-commands', daemon=True)
        self.engine = None
        self.service = None
        self.stream_id = 0
        self._last_token = None
        self._had_signal = False
        self._command_busy = False
        self._calibrating = False
        self._speech = True
        self._preview_difference = False
        self._tick_task = None
        self._closing = False
        self._thread.start()

    def _emit(self, kind, **data):
        try:
            self.send({'type': kind, **data})
        except Exception:
            LOGGER.exception('Desktop transport failed')

    async def _io(self, function, *args):
        return await self._loop.run_in_executor(self._vision, function, *args)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._initialize())
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    async def _initialize(self):
        try:
            try:
                settings = self.store.load()
            except Exception as exc:
                settings = RobotSettings()
                self._emit('error', message=f'配置读取失败，使用默认模拟配置：{exc}')
            settings.backend = 'mock'
            self.service = RobotService(settings, self.store)
            self.engine = await self._io(RecognitionEngine, self.config)
            self._emit('ready', hardware_opened=False, model=self.engine.model_status)
            self.ready.set()
            self._tick_task = asyncio.create_task(self._tick())
        except Exception as exc:
            self._emit('error', message=f'初始化失败：{exc}')
            self.ready.set()

    def handle(self, message):
        if self.closed or self._closing:
            return None
        if not isinstance(message, dict) or message.get('type', 'command') != 'command':
            self._emit('error', message='无效桌面命令')
            return None
        return asyncio.run_coroutine_threadsafe(self._execute(dict(message)), self._loop)

    def _sensor_ready(self):
        packet = camera_manager.frame_packet()
        return bool(self.engine and self.engine.model and self.engine.detector.calibrated
                    and not self._calibrating and packet is not None
                    and packet.serial == self.stream_id
                    and time.monotonic() - packet.timestamp < self.config.recognition.max_frame_age)

    async def _execute(self, message):
        name = message.get('name')
        if not self.ready.is_set() or self.service is None or self.engine is None:
            if name == 'ready':
                self._emit('status', phase='initializing', message='正在加载模型', busy=True,
                           connected=False, teaching=False, points={}, backend='mock')
                return
            self._emit('error', message='模型正在初始化，请稍候')
            return
        if name == 'close':
            # Transport owner should also call close() after handling the event.
            self._emit('close_requested')
            return
        exceptional = name in ('stop', 'speech')
        if self._command_busy and not exceptional:
            self._emit('error', message='当前操作尚未结束')
            return
        if not exceptional:
            self._command_busy = True
        s = self.service
        try:
            if name == 'ready':
                self._status()
            elif name == 'connect':
                if s.arm is not None:
                    raise ValueError('设备已连接')
                backend = message.get('backend', 'mock')
                if backend == 'nero' and message.get('real_confirmed') is not True:
                    raise ValueError('真机需先完成固件、夹爪、负载与路径现场确认')
                settings = replace(s.settings, backend=backend, channel=str(message.get('channel', '0')))
                settings.validate()
                self.store.save(settings)
                s.settings = settings
                await s.connect(self._owner)
            elif name == 'disconnect':
                await s.disconnect(self._owner)
            elif name == 'teach_start':
                await s.teach(self._owner, True)
            elif name == 'teach_end':
                await s.teach(self._owner, False)
            elif name == 'save_point':
                await s.save_point(self._owner, message.get('point'))
            elif name == 'mock_drag':
                s.require_owner(self._owner)
                if not s.mock or not s.teaching:
                    raise ValueError('仅离线示教可设置模拟姿态')
                await s.call('simulate_drag', joints(message.get('joints')))
            elif name == 'detect':
                ready = s.mock or self._sensor_ready()
                await s.start(self._owner, ready=ready, stream_id=self.stream_id,
                              mock_key=message.get('key', 'rat'), on_result=self._result,
                              ready_check=self._sensor_ready)
            elif name == 'stop':
                await s.stop(self._owner)
            elif name == 'settings':
                if s.busy or s.teaching:
                    raise ValueError('请结束当前运动或示教')
                allowed = {'speed_percent', 'open_width', 'close_width', 'gripper_force'}
                values = message.get('values', {})
                if not isinstance(values, dict) or set(values) - allowed:
                    raise ValueError('无效夹爪或速度设置')
                values = {key: int(value) if key == 'speed_percent' else float(value) for key, value in values.items()}
                settings = replace(s.settings, **values)
                settings.validate()
                self.store.save(settings)
                s.settings = settings
                if s.arm:
                    s.arm.settings = settings
            elif name in ('scan_cameras', 'connect_camera', 'disconnect_camera', 'calibrate', 'reload_model'):
                if s.busy or s.teaching:
                    raise ValueError('机械臂运动或示教时不能更改传感器/模型')
                await self._sensor_command(name, message)
            elif name == 'speech':
                self._speech = bool(message.get('enabled', True))
                if not self._speech:
                    self._silence()
            elif name == 'test_speech':
                self._play(message.get('key', 'rat'))
            elif name == 'preview':
                self._preview_difference = bool(message.get('difference', False))
            else:
                raise ValueError(f'未知命令：{name}')
            self._emit('ack', name=name)
        except Exception as exc:
            LOGGER.exception('Desktop command %s failed', name)
            self._emit('error', message=str(exc), command=name)
        finally:
            if not exceptional:
                self._command_busy = False
            self._status()

    async def _sensor_command(self, name, message):
        if name == 'scan_cameras':
            cameras = await self._io(lambda: camera_manager.scan(max_index=self.config.sensor.scan_max_index))
            await self._io(self.engine.reset)
            self.stream_id, self._last_token = 0, None
            self._emit('cameras', items=[{'index': c.index, 'name': c.display_name, 'available': c.available} for c in cameras])
        elif name == 'connect_camera':
            index = int(message.get('index', 0))
            camera = next((c for c in camera_manager.cameras if c.index == index and c.available), None)
            if camera is None:
                raise ValueError('请先扫描并选择可用传感器')
            stream = await self._io(lambda: camera_manager.select(camera, owner=self._owner,
                              target_fps=self.config.sensor.capture_fps, encode_jpeg=False))
            if not await self._io(stream.wait_ready):
                raise ValueError(stream.error or '摄像头启动失败')
            await self._io(self.engine.reset)
            self.stream_id, self._last_token = stream.serial, None
        elif name == 'disconnect_camera':
            await self._io(camera_manager.close)
            await self._io(self.engine.reset)
            self.stream_id, self._last_token = 0, None
        elif name == 'reload_model':
            await self._io(self.engine.reload_model, message.get('path'))
        elif name == 'calibrate':
            self._calibrating = True
            try:
                frames, sequence = [], None
                serial = self.stream_id
                deadline = time.monotonic() + 12
                while len(frames) < self.config.contact.calibration_frames:
                    if self._closing:
                        raise RuntimeError('程序正在退出')
                    p = camera_manager.frame_packet()
                    if p is not None:
                        if p.serial != serial:
                            raise ValueError('传感器已重连，请重新校准')
                        if p.sequence != sequence and time.monotonic()-p.timestamp < self.config.recognition.max_frame_age:
                            frames.append(p.frame)
                            sequence = p.sequence
                    if time.monotonic() > deadline:
                        raise TimeoutError('空载校准未收到足够的新鲜图像')
                    await asyncio.sleep(.04)
                await self._io(self.engine.calibrate, frames)
                p = camera_manager.frame_packet()
                if p is None or p.serial != serial or time.monotonic()-p.timestamp > self.config.recognition.max_frame_age:
                    await self._io(self.engine.reset)
                    raise RuntimeError('校准期间传感器信号中断')
                self._emit('notice', message='空载校准完成')
            finally:
                self._calibrating = False

    def _result(self, key, state, simulated):
        zodiac = ZODIAC_BY_KEY.get(key)
        p = state.prediction if state else None
        self._emit('result', key=key, glyph=zodiac.glyph if zodiac else '—',
                   title=zodiac.title if zodiac else '本轮未识别', simulated=simulated,
                   confidence=p.confidence if p else None, poem=zodiac.poem if zodiac else '')
        if key and self._speech:
            self._play(key)

    def _play(self, key):
        if key not in ZODIAC_BY_KEY:
            return
        try:
            import winsound
            path = PROJECT_ROOT / 'assets' / 'speech' / f'{key}.wav'
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except Exception as exc:
            self._emit('error', message=f'语音播放失败：{exc}')

    def _silence(self):
        try:
            import winsound
            winsound.PlaySound(None, 0)
        except Exception:
            pass

    def _status(self):
        s = self.service
        if s is None:
            return
        packet = camera_manager.frame_packet()
        sensor_connected = bool(packet and time.monotonic()-packet.timestamp < self.config.recognition.max_frame_age)
        self._emit('status', phase=s.phase, message=s.message, busy=s.busy or self._command_busy,
                   teaching=s.teaching, connected=s.arm is not None, backend=s.settings.backend,
                   points=dict(s.points), model=self.engine.model_status if self.engine else '正在加载',
                   sensor=sensor_connected, sensor_ready=self._sensor_ready(), calibrated=bool(self.engine and self.engine.detector.calibrated),
                   calibrating=self._calibrating, cycle=s.cycle_id, cycle_id=s.cycle_id, completed=s.completed, speech=self._speech,
                   settings={k: getattr(s.settings, k) for k in ('channel','speed_percent','open_width','close_width','gripper_force')})

    async def _tick(self):
        while not self._closing:
            try:
                packet = camera_manager.frame_packet()
                fresh = bool(packet and time.monotonic()-packet.timestamp < self.config.recognition.max_frame_age)
                if not fresh and self._had_signal:
                    await self._io(self.engine.reset)
                    self._had_signal = False
                elif fresh:
                    self._had_signal = True
                if packet is not None and (packet.serial, packet.sequence) != self._last_token:
                    self._last_token = packet.serial, packet.sequence
                    if packet.serial != self.stream_id:
                        await self._io(self.engine.reset)
                        self.stream_id = packet.serial
                    def process_frame():
                        import cv2
                        frame = packet.frame
                        if self._preview_difference and self.engine.detector.baseline is not None:
                            from core.contact import make_difference
                            frame = make_difference(frame, self.engine.detector.baseline)
                        h, w = frame.shape[:2]
                        if w > 900:
                            frame = cv2.resize(frame, (900, round(h*900/w)))
                        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                        state = None
                        if not self._calibrating and not self.service.mock and self.service.phase in ('recognizing','releasing'):
                            state = self.engine.process(packet.frame, sequence=packet.sequence,
                                                        timestamp=packet.timestamp, stream_id=packet.serial)
                        return encoded.tobytes() if ok else None, state
                    jpeg, state = await self._io(process_frame)
                    if jpeg:
                        self._emit('frame', jpeg=base64.b64encode(jpeg).decode('ascii'))
                    if state:
                        self.service.observe(state, packet.timestamp, packet.serial)
                        self._emit('contact', active=state.contact.active, stable=state.contact.stable,
                                   score=state.contact.score, message=state.message)
                self._status()
            except Exception as exc:
                LOGGER.exception('Desktop update failed')
                self._emit('error', message=str(exc))
            await asyncio.sleep(.1)

    async def _shutdown(self):
        self._closing = True
        if self._tick_task:
            self._tick_task.cancel()
            await asyncio.gather(self._tick_task, return_exceptions=True)
        if self.service and self.service.arm is not None:
            try:
                await self.service.disconnect(self._owner)
            except Exception as exc:
                self._emit('error', message=f'设备关闭异常：{exc}')
        await self._io(camera_manager.close_if_owner, self._owner)
        self._silence()
        if self.service:
            self.service.pool.shutdown(wait=True)
        self._emit('closed')

    def close(self):
        if self.closed:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            future.result(timeout=25)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._vision.shutdown(wait=True, cancel_futures=True)
            self.closed = True
