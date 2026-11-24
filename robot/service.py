"""One process-wide arm owner, serialized SDK access, one cycle per start."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from robot.backend import MockArm
from robot.settings import joints


class Stopped(Exception):
    pass


class RobotService:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.arm = None
        self.owner = None
        self.phase = 'disconnected'
        self.message = '未连接；默认离线模拟'
        self.busy = False
        self.teaching = False
        self.cycle_id = 0
        self.completed = 0
        self.result = None
        self.recognition_start = 0.
        self.observation = None
        self.stop_requested = False
        self.task = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nero')
        self.monitor_task = None
        self.last_snapshot = None

    async def call(self, name, *args):
        if self.arm is None:
            raise RuntimeError('请先连接')
        if self.stop_requested and name in ('move', 'gripper', 'enable', 'enter_teach'):
            raise Stopped('已请求停止')
        return await asyncio.get_running_loop().run_in_executor(self.pool, getattr(self.arm, name), *args)

    def require_owner(self, owner):
        if self.owner is not owner:
            raise RuntimeError('机械臂由另一个窗口占用')

    async def connect(self, owner):
        if self.owner is not None:
            raise RuntimeError('机械臂已有控制窗口')
        self.owner = owner
        self.busy = True
        try:
            self.settings.validate()
            if self.settings.backend == 'mock':
                self.arm = MockArm(self.settings)
            else:
                from robot.nero import NeroArm
                self.arm = NeroArm(self.settings)
            await self.call('connect')
            self.phase, self.message = 'idle', '模拟设备已连接' if self.mock else 'NERO 已连接；未自动使能'
            self.monitor_task = asyncio.create_task(self._monitor())
        except Exception:
            if self.arm:
                await self.call('disconnect')
            self.owner, self.arm = None, None
            raise
        finally:
            self.busy = False

    @property
    def mock(self):
        return isinstance(self.arm, MockArm)

    @property
    def points(self):
        return self.settings.points.get(self.arm.profile, {}) if self.arm else {}

    async def _monitor(self):
        while self.arm is not None:
            # Mode transitions can temporarily suppress telemetry; transition
            # methods perform their own bounded verification.
            if self.phase in ('idle', 'teaching', 'complete') and not self.busy:
                try:
                    self.last_snapshot = await self.call('snapshot')
                except Exception as exc:
                    self.phase, self.message = 'fault', str(exc)
                    self.stop_requested = True
            await asyncio.sleep(.1)

    async def teach(self, owner, enter):
        self.require_owner(owner)
        if self.busy or self.phase == 'fault':
            raise RuntimeError('当前状态不能示教；故障后请断开重连')
        self.busy = True
        self.phase = 'switching'
        try:
            if enter:
                await self.call('enable')
                await self.call('gripper', self.settings.open_width)
                await self.wait_for(lambda s: abs(s.width-self.settings.open_width) < .002, 5)
                await self.call('enter_teach')
            else:
                await self.call('exit_teach')
            await self.wait_for(lambda s: s.teaching == enter, 3)
            self.teaching = enter
            self.phase = 'teaching' if enter else 'idle'
            self.message = ('模拟重力补偿示教' if self.mock else '原生拖动示教：请扶稳机械臂') if enter else '已退出示教，保持当前位置'
        except Exception as exc:
            self.phase, self.message = 'fault', str(exc)
            raise
        finally:
            self.busy = False

    async def save_point(self, owner, name):
        self.require_owner(owner)
        if self.busy or self.phase != 'teaching' or name not in ('detect', 'home'):
            raise RuntimeError('请在示教模式保存检测点或收回点')
        self.busy = True
        try:
            first = await self.call('snapshot')
            await asyncio.sleep(.35)
            second = await self.call('snapshot')
            if max(abs(a-b) for a,b in zip(first.q, second.q)) > .01:
                raise ValueError('机械臂仍在移动，请保持静止后再保存')
            profile = self.arm.profile
            previous = self.settings.points.get(profile, {}).copy()
            self.settings.points.setdefault(profile, {})[name] = list(joints(second.q))
            try:
                self.store.save(self.settings)
            except Exception:
                self.settings.points[profile] = previous
                raise
            self.message = ('检测点' if name == 'detect' else '收回点') + '已保存至应用配置'
        finally:
            self.busy = False

    def observe(self, state, timestamp, stream_id):
        self.observation = (state, timestamp, stream_id)

    def start(self, owner, *, ready, stream_id=0, mock_key='rat', on_result=None, ready_check=None):
        self.require_owner(owner)
        if self.busy or self.teaching or self.phase not in ('idle', 'complete'):
            raise RuntimeError('请结束示教，并等待当前一轮结束')
        if set(self.points) != {'detect', 'home'}:
            raise RuntimeError('请先保存检测点和收回点')
        if not ready:
            raise RuntimeError('传感器、模型或空载校准尚未就绪')
        self.settings.validate()
        self.busy = True  # Set synchronously: double clicks cannot queue a cycle.
        self.stop_requested = False
        self.result = None
        self.observation = None
        self.ready_check = ready_check
        self.cycle_id += 1
        self.phase, self.message = 'preparing', '开始单次检测'
        self.task = asyncio.create_task(self._cycle(dict(self.points), stream_id, mock_key, on_result))
        return self.task

    async def wait_for(self, predicate, timeout, *, stable=None):
        deadline = time.monotonic() + timeout
        since = None
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise Stopped('已请求停止')
            s = await self.call('snapshot')
            self.last_snapshot = s
            now = time.monotonic()
            if now - s.timestamp > .5:
                raise RuntimeError('机械臂反馈过期')
            if predicate(s):
                since = now if since is None else since
                if now - since >= (self.settings.stable_seconds if stable is None else stable):
                    return s
            else:
                since = None
            await asyncio.sleep(.05)
        raise TimeoutError('等待到位或反馈超时')

    async def _move(self, q):
        if not self.mock and self.ready_check and not self.ready_check():
            raise RuntimeError('传感器已断流，禁止继续运动')
        await self.call('move', q)
        await self.wait_for(lambda s: max(abs(a-b) for a,b in zip(s.q, q)) < self.settings.joint_tolerance,
                            self.settings.move_timeout)

    def _observation(self, stream_id):
        if self.observation is None:
            return None
        state, ts, serial = self.observation
        if serial != stream_id or time.monotonic()-ts > .5:
            raise RuntimeError('触觉信号中断或相机发生切换')
        if state.status in ('no_signal', 'uncalibrated', 'error', 'model_missing'):
            raise RuntimeError('触觉检测不可用：' + state.message)
        return state, ts

    async def _recognize(self, stream_id, mock_key):
        from core.zodiac import ZODIAC_BY_KEY
        self.recognition_start = time.monotonic()
        candidate, since, last_ts = None, 0., 0.
        while time.monotonic()-self.recognition_start < self.settings.recognize_timeout:
            if self.stop_requested:
                raise Stopped('已请求停止')
            await self.call('snapshot')
            if self.mock:
                if time.monotonic()-self.recognition_start >= .8 and mock_key in ZODIAC_BY_KEY:
                    return mock_key, None
            else:
                obs = self._observation(stream_id)
                if obs is None:
                    if time.monotonic()-self.recognition_start > .5:
                        raise RuntimeError('未收到触觉检测帧')
                else:
                    state, ts = obs
                    p = state.prediction
                    if ts > self.recognition_start and ts > last_ts:
                        last_ts = ts
                        key = p.key if (p and p.accepted and state.contact.active and state.contact.stable
                                        and state.status in ('preview','recognized')) else None
                        if key not in ZODIAC_BY_KEY:
                            candidate = None
                        elif key != candidate:
                            candidate, since = key, ts
                        elif ts-since >= .8:
                            return key, state
            await asyncio.sleep(.05)
        return None, None

    async def _release(self, stream_id):
        self.phase, self.message = 'releasing', '张开夹爪，确认接触解除'
        started = time.monotonic()
        await self.call('gripper', self.settings.open_width)
        def released(s):
            if abs(s.width-self.settings.open_width) >= .002:
                return False
            if self.mock:
                return True
            obs = self._observation(stream_id)
            return bool(obs and obs[1] > started and obs[0].status == 'no_contact')
        await self.wait_for(released, self.settings.release_timeout)

    async def _cycle(self, points, stream_id, mock_key, on_result):
        try:
            await self.call('enable')
            await self.call('gripper', self.settings.open_width)
            await self.wait_for(lambda s: abs(s.width-self.settings.open_width) < .002, 5)
            self.phase, self.message = 'approaching', '移动至检测点'
            await self._move(points['detect'])
            if not self.mock and self.ready_check and not self.ready_check():
                raise RuntimeError('传感器已断流，禁止闭合')
            self.phase, self.message = 'recognizing', '闭合压牌，等待稳定识别'
            await self.call('gripper', self.settings.close_width)
            key, state = await self._recognize(stream_id, mock_key)
            self.result = (key, state, self.mock)
            if on_result:
                on_result(key, state, self.mock)
            await self._release(stream_id)
            self.phase, self.message = 'returning', '接触已解除，返回收回点'
            await self._move(points['home'])
            self.completed += 1
            self.phase = 'complete'
            self.message = '本轮结束，等待再次点击检测' if key else '本轮未识别，已松开并收回'
        except Exception as exc:
            self.phase = 'fault'
            self.message = str(exc) + '；未自动收回，请检查后重连'
            try:
                await self.call('hold')
            except Exception:
                self.message += '；保持指令失败，请使用硬件急停'
        finally:
            self.busy = False

    async def stop(self, owner):
        self.require_owner(owner)
        self.stop_requested = True
        if self.task and not self.task.done():
            await self.task
        else:
            try:
                await self.call('hold')
            finally:
                self.phase, self.message = 'fault', '已停止；检查后断开重连'
                self.teaching = False

    async def disconnect(self, owner):
        self.require_owner(owner)
        try:
            await self.stop(owner)
        finally:
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            await self.call('disconnect')
            self.arm, self.owner = None, None
            self.busy = self.teaching = False
            self.stop_requested = False
            self.phase, self.message = 'disconnected', '已断开'
