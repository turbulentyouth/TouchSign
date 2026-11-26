"""One process-wide arm owner, serialized SDK access, one cycle per start."""
import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor
import time
from robot.backend import ArmNeedsEnable, MockArm
from robot.settings import joints
from core.rejection import PREVIEW_CONFIRM_FRAMES, PREVIEW_CONFIRM_SECONDS

LOGGER = logging.getLogger(__name__)

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
        self.recognition_confidence_threshold = .5
        self.observation = None
        self.ready_check = None
        self.stop_requested = False
        self.finishing = False
        self._cycle_stream_id = 0
        self.task = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nero')
        self.monitor_task = None
        self.last_snapshot = None
        self._disconnect_lock = asyncio.Lock()

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
        self.observation = None
        self.ready_check = None
        self._cycle_stream_id = 0
        self.phase, self.message = 'connecting', f'正在连接 {self.settings.backend}：{self.settings.channel}'
        LOGGER.info('机械臂连接开始：backend=%s, channel=%s', self.settings.backend, self.settings.channel)
        try:
            self.settings.validate()
            if self.settings.backend == 'mock':
                self.arm = MockArm(self.settings)
            else:
                from robot.nero import NeroArm
                self.arm = NeroArm(self.settings)
            await self.call('connect')
            self.phase, self.message = 'idle', '模拟设备已连接' if self.mock else 'NERO 已连接；未自动使能'
            LOGGER.info('机械臂连接成功：backend=%s, channel=%s', self.settings.backend, self.settings.channel)
            self.monitor_task = asyncio.create_task(self._monitor())
        except Exception as exc:
            self.phase, self.message = 'disconnected', f'连接失败：{exc}'
            LOGGER.exception('机械臂连接失败：backend=%s, channel=%s', self.settings.backend, self.settings.channel)
            try:
                if self.arm:
                    await self.call('disconnect')
            except Exception:
                LOGGER.exception('连接失败后关闭机械臂通信异常')
            finally:
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
            if self.phase in ('idle', 'disabled', 'teaching', 'complete') and not self.busy:
                try:
                    snapshot = await self.call('snapshot')
                    if not self.busy and not self.stop_requested:
                        self.last_snapshot = snapshot
                        if self.phase == 'disabled':
                            self.phase, self.message = 'idle', 'NERO 状态正常'
                except ArmNeedsEnable as exc:
                    if not self.busy and not self.stop_requested:
                        self.last_snapshot = None
                        self.phase, self.message = 'disabled', str(exc)
                except Exception as exc:
                    if not self.busy and not self.stop_requested:
                        self.phase, self.message = 'fault', str(exc)
                        self.stop_requested = True
            await asyncio.sleep(.1)

    async def teach(self, owner, enter):
        self.require_owner(owner)
        allowed = ('idle', 'disabled', 'complete') if enter else ('teaching',)
        if self.busy or self._disconnect_lock.locked() or self.phase not in allowed:
            raise RuntimeError('当前状态不能示教；故障后请断开重连')
        self.busy = True
        self.phase = 'switching'
        self.message = '准备使能并进入示教' if enter else '正在退出示教，等待位置保持反馈'
        LOGGER.info('示教模式切换：enter=%s, channel=%s', enter, self.settings.channel)
        try:
            if enter:
                await self.call('enable')
                await self.call('gripper', self.settings.open_width)
                await self.wait_for(lambda s: abs(s.width-self.settings.open_width) < .002, 5)
                self.message = '正在切换示教，等待新的关节反馈'
                await self.call('enter_teach')
            else:
                await self.call('exit_teach')
            await self.wait_for(lambda s: s.teaching == enter, 3)
            self.teaching = enter
            self.phase = 'teaching' if enter else 'idle'
            self.message = ('模拟重力补偿示教' if self.mock else '原生拖动示教：请扶稳机械臂') if enter else '已退出示教，保持当前位置'
            LOGGER.info('示教模式切换完成：phase=%s', self.phase)
        except Exception as exc:
            self.phase, self.message = 'fault', str(exc)
            LOGGER.exception('示教模式切换失败：enter=%s', enter)
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

    def start(self, owner, *, ready, stream_id=0, mock_key='rat', on_result=None, ready_check=None,
              confidence_threshold=.5):
        self.require_owner(owner)
        if self.busy or self._disconnect_lock.locked() or self.teaching or self.phase not in ('idle', 'complete'):
            raise RuntimeError('请结束示教，并等待当前一轮结束')
        if set(self.points) != {'detect', 'home'}:
            raise RuntimeError('请先保存检测点和收回点')
        if not ready:
            raise RuntimeError('传感器、模型或空载校准尚未就绪')
        if not math.isfinite(confidence_threshold) or not 0 < confidence_threshold <= 1:
            raise ValueError('识别置信度门槛必须在 (0, 1]')
        self.settings.validate()
        self.recognition_confidence_threshold = confidence_threshold
        self.busy = True  # Set synchronously: double clicks cannot queue a cycle.
        self.stop_requested = False
        self.result = None
        self.observation = None
        self.ready_check = ready_check
        self._cycle_stream_id = stream_id
        self.cycle_id += 1
        self.phase, self.message = 'preparing', '开始单次检测'
        LOGGER.info('检测开始：cycle=%s, channel=%s, 置信度门槛=%.0f%%',
                    self.cycle_id, self.settings.channel, 100*self.recognition_confidence_threshold)
        self.task = asyncio.create_task(self._cycle(dict(self.points), stream_id, mock_key, on_result))
        return self.task

    async def wait_for(self, predicate, timeout, *, stable=None):
        deadline = time.monotonic() + timeout
        since = None
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise Stopped('已请求停止')
            s = await self.call('snapshot')
            if self.stop_requested:
                raise Stopped('已请求停止')
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

    def _observation(self, stream_id, *, after=None):
        if self.observation is None:
            return None
        state, ts, serial = self.observation
        if not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise RuntimeError('触觉帧时间戳无效')
        # A phase starts before its vision worker supplies its first new frame.
        # Results captured earlier (including a late worker result) are not
        # evidence of either release or a failure of the current stream.
        if after is not None and ts <= after:
            return None
        age = time.monotonic() - ts
        if serial != stream_id:
            raise RuntimeError(f'相机发生切换：预期流 {stream_id}，收到流 {serial}；请重新空载校准')
        if not -.1 <= age <= .5:
            raise RuntimeError(f'触觉信号中断：帧龄 {age:.3f} 秒，允许范围 -0.1–0.5 秒')
        if state.status in ('no_signal', 'uncalibrated', 'error', 'model_missing'):
            raise RuntimeError('触觉检测不可用：' + state.message)
        return state, ts

    async def _grip(self):
        target = self.settings.close_width
        speed = self.settings.gripper_close_speed
        tolerance = .001  # 1 mm, with the configured continuous arrival interval.
        self.phase, self.message = 'gripping', f'夹爪闭合至 {target*1000:g} mm，限速 {speed*1000:g} mm/s'
        LOGGER.info('夹爪闭合：cycle=%s, 目标间距=%.3f mm, 指令限速=%.1f mm/s',
                    self.cycle_id, target*1000, speed*1000)

        def check(snapshot):
            if self.stop_requested:
                raise Stopped('已请求停止')
            if not self.mock and self.ready_check and not self.ready_check():
                raise RuntimeError('传感器已断流，停止闭合')
            if (not math.isfinite(snapshot.timestamp)
                    or not -.1 <= time.monotonic() - snapshot.timestamp <= .5
                    or not math.isfinite(snapshot.width) or not 0 <= snapshot.width <= .1):
                raise RuntimeError('夹爪开口反馈无效或过期，停止闭合')
            if snapshot.teaching:
                raise RuntimeError('示教中不能闭合夹爪')
            self.last_snapshot = snapshot

        def arrived(snapshot):
            check(snapshot)
            self.message = (f'等待夹爪到位：当前 {snapshot.width*1000:.3f} mm，'
                            f'目标 {target*1000:.3f} mm')
            return abs(snapshot.width - target) <= tolerance

        if self.stop_requested:
            raise Stopped('已请求停止')
        snapshot = await self.call('snapshot')
        check(snapshot)
        # Allow travel at the chosen speed, plus the existing arrival allowance.
        duration = max(0., snapshot.width-target) / speed
        deadline = time.monotonic() + duration + self.settings.grip_timeout
        pending = False
        try:
            try:
                if self.mock or snapshot.width <= target:
                    # Offline demonstrations compress the travel time. Opening
                    # to a wider target does not need the closing-only ramp.
                    pending = True
                    await self.call('gripper', target)
                else:
                    # The AGX width command has no velocity field. Slew its
                    # position setpoint without running tactile inference.
                    commanded = min(snapshot.width, self.settings.open_width)
                    sent_at = time.monotonic()
                    period, max_lead = .05, .002
                    while commanded > target:
                        await asyncio.sleep(period)
                        snapshot = await self.call('snapshot')
                        check(snapshot)
                        now = time.monotonic()
                        self.message = (f'夹爪闭合：当前 {snapshot.width*1000:.3f} mm，'
                                        f'目标 {target*1000:.3f} mm，限速 {speed*1000:g} mm/s')
                        if now >= deadline:
                            raise TimeoutError('闭合行程超时')
                        # Never catch up with a large jump after a scheduling
                        # delay, or accumulate a far-away target when stalled.
                        step = speed * min(period, max(0., now-sent_at))
                        next_width = max(target, commanded-step, snapshot.width-max_lead)
                        if next_width >= commanded:
                            continue
                        pending = True
                        await self.call('gripper', next_width)
                        commanded, sent_at = next_width, time.monotonic()
                snapshot = await self.wait_for(arrived, max(0., deadline-time.monotonic()))
            except TimeoutError as exc:
                raise TimeoutError(f'夹爪到位超时（计划行程 {duration:.1f} 秒 + '
                                   f'等待余量 {self.settings.grip_timeout:g} 秒）：'
                                   f'{self.message}；已停止闭合，不继续收紧或加力') from exc
            pending = False  # Keep the reached target during recognition.
            self.observation = None
            LOGGER.info('夹爪已到位：cycle=%s, 目标=%.3f mm, 实际=%.3f mm；开始触觉检测',
                        self.cycle_id, target*1000, snapshot.width*1000)
        finally:
            if pending:
                try:
                    await self.call('hold_gripper')
                except Exception as exc:
                    raise RuntimeError('无法确认夹爪保持，可能仍有未完成的闭合指令；请使用硬件急停') from exc

    async def _recognize(self, stream_id, mock_key):
        from core.zodiac import ZODIAC_BY_KEY
        self.recognition_start = time.monotonic()
        candidate, since, last_ts, count = None, 0., 0., 0
        while time.monotonic()-self.recognition_start < self.settings.recognize_timeout:
            if self.stop_requested:
                raise Stopped('已请求停止')
            await self.call('snapshot')
            if self.mock:
                if time.monotonic()-self.recognition_start >= .8 and mock_key in ZODIAC_BY_KEY:
                    return mock_key, None
            else:
                obs = self._observation(stream_id, after=self.recognition_start)
                if obs is None:
                    if time.monotonic()-self.recognition_start > .5:
                        raise RuntimeError('未收到触觉检测帧')
                else:
                    state, ts = obs
                    p = state.prediction
                    if ts > self.recognition_start and ts > last_ts:
                        if ts-last_ts > .5:
                            candidate, count = None, 0
                        last_ts = ts
                        key = p.key if (p and p.accepted and state.contact.calibrated and state.contact.valid
                                        and state.contact.active and state.contact.stable
                                        and state.status in ('preview','recognized')) else None
                        if key in ZODIAC_BY_KEY and state.status == 'preview':
                            confidence = p.confidence
                            if not math.isfinite(confidence) or confidence < self.recognition_confidence_threshold:
                                self.message = (f'置信度不足：{ZODIAC_BY_KEY[key].glyph} {confidence:.0%}，'
                                                f'需达到 {self.recognition_confidence_threshold:.0%}；保持接触继续识别')
                                key = None
                        if key not in ZODIAC_BY_KEY:
                            candidate, count = None, 0
                        else:
                            if key != candidate:
                                candidate, since, count = key, ts, 1
                            else:
                                count += 1
                            self.message = f'正在确认{ZODIAC_BY_KEY[key].glyph}：连续 {count} 帧'
                            confirmed = (state.status == 'recognized' or
                                         (count >= PREVIEW_CONFIRM_FRAMES
                                          and ts-since >= PREVIEW_CONFIRM_SECONDS-1e-9))
                            if confirmed:
                                LOGGER.info('触觉结果确认：cycle=%s, key=%s, mode=%s, '
                                            '置信度=%.1f%%, 到位后耗时=%.3f 秒, 确认帧=%s', self.cycle_id,
                                            key, state.status, 100*p.confidence,
                                            time.monotonic()-self.recognition_start, count)
                                return key, state
            await asyncio.sleep(.05)
        LOGGER.info('触觉识别超时：cycle=%s, %s', self.cycle_id, self.message)
        return None, None

    async def _release(self, stream_id):
        self.phase, self.message = 'releasing', '张开夹爪，确认接触解除'
        started = time.monotonic()
        self.observation = None
        LOGGER.info('检测进入松开阶段：cycle=%s, stream=%s；等待本次松开后的新触觉帧',
                    self.cycle_id, stream_id)
        await self.call('gripper', self.settings.open_width)
        def released(s):
            if abs(s.width-self.settings.open_width) >= .002:
                self.message = (f'等待夹爪张开：当前 {s.width*1000:.1f} mm，'
                                f'目标 {self.settings.open_width*1000:.1f} mm')
                return False
            if self.mock:
                return True
            obs = self._observation(stream_id, after=started)
            if not obs:
                self.message = '夹爪已张开，等待新的触觉反馈后确认接触解除'
                return False
            if obs[0].status != 'no_contact':
                self.message = '夹爪已张开，传感器尚未确认无接触；确认后自动收回'
                return False
            self.message = '接触已解除，确认稳定后自动收回'
            return True
        try:
            await self.wait_for(released, self.settings.release_timeout)
        except TimeoutError as exc:
            raise TimeoutError(f'松开确认超时（{self.settings.release_timeout:g} 秒）：{self.message}') from exc

    async def _cycle(self, points, stream_id, mock_key, on_result):
        try:
            await self.call('enable')
            await self.call('gripper', self.settings.open_width)
            await self.wait_for(lambda s: abs(s.width-self.settings.open_width) < .002, 5)
            self.phase, self.message = 'approaching', '移动至检测点'
            LOGGER.info('检测进入检测点运动阶段：cycle=%s', self.cycle_id)
            await self._move(points['detect'])
            if not self.mock and self.ready_check and not self.ready_check():
                raise RuntimeError('传感器已断流，禁止闭合')
            await self._grip()
            self.phase, self.message = 'recognizing', '夹爪已到位，保持目标间距并进行触觉识别'
            key, state = await self._recognize(stream_id, mock_key)
            if self.stop_requested:
                raise Stopped('已请求停止')
            self.result = (key, state, self.mock)
            if on_result:
                on_result(key, state, self.mock)
            LOGGER.info('本轮识别结束：cycle=%s, result=%s；准备松开并收回', self.cycle_id, key)
            await self._release(stream_id)
            self.phase, self.message = 'returning', '接触已解除，返回收回点'
            LOGGER.info('接触解除已确认，开始收回：cycle=%s', self.cycle_id)
            await self._move(points['home'])
            self.completed += 1
            self.phase = 'complete'
            self.message = '本轮结束，等待再次点击检测' if key else '本轮未识别，已松开并收回'
            LOGGER.info('检测完成并已收回：cycle=%s', self.cycle_id)
        except Exception as exc:
            if isinstance(exc, Stopped) and self.finishing:
                await self._finish_stopped_cycle(points['home'], stream_id)
                return
            failed_phase = self.phase
            if isinstance(exc, Stopped):
                self.phase = 'stopped'
                self.message = '已停止本轮；后续收回已取消，检查后请断开重连'
                LOGGER.info('检测已停止：cycle=%s, phase=%s', self.cycle_id, failed_phase)
            else:
                self.phase = 'fault'
                self.message = str(exc) + '；未自动收回，请检查后重连'
                LOGGER.exception('检测失败：cycle=%s, phase=%s', self.cycle_id, failed_phase)
            try:
                await self.call('hold')
            except Exception:
                self.phase = 'fault'
                self.message += '；保持指令失败，请使用硬件急停'
                LOGGER.exception('检测中止后保持失败：cycle=%s', self.cycle_id)
        finally:
            self.busy = False
            self.finishing = False

    async def stop_and_return(self, owner, *, stream_id=None, ready_check=None):
        """User button: finish the current round by releasing, then returning."""
        self.require_owner(owner)
        if self._disconnect_lock.locked():
            raise RuntimeError('正在断开连接，不能收回')
        if self.finishing:
            # Repeated clicks join the same operation, never queue another move.
            await asyncio.shield(self.task)
            return
        if self.teaching or self.phase in ('disconnected', 'disabled', 'teaching', 'switching'):
            raise RuntimeError('请先完成使能和示教，再停止本轮并收回')
        running = self.task is not None and not self.task.done()
        if self.busy and not running:
            raise RuntimeError('当前操作尚未结束，不能收回')
        if 'home' not in self.points:
            raise RuntimeError('请先保存收回点')
        home = joints(self.points['home'])
        self.settings.validate()
        LOGGER.info('收到停止并收回请求：cycle=%s, phase=%s', self.cycle_id, self.phase)
        self.busy = self.finishing = True
        self.stop_requested = True
        self.message = '正在结束本轮检测，随后张开夹爪并收回'
        if not running:
            if ready_check is not None:
                self.ready_check = ready_check
            serial = self._cycle_stream_id if stream_id is None else stream_id
            self.task = asyncio.create_task(self._finish_stopped_cycle(home, serial))
        # A running cycle first unwinds its closing operation and cancels the
        # outstanding gripper target, then takes this same finishing path.
        await asyncio.shield(self.task)

    async def _finish_stopped_cycle(self, home, stream_id):
        opening = False
        try:
            self.phase, self.message = 'preparing', '准备停止并收回，校验机械臂状态和收回点限位'
            await self.call('hold')
            snapshot = await self.call('snapshot')
            if snapshot.teaching or not math.isfinite(snapshot.timestamp) or not -.1 <= time.monotonic()-snapshot.timestamp <= .5:
                raise RuntimeError('机械臂模式或反馈未确认，不能收回')
            if not self.finishing:
                raise Stopped('收回已中止')
            # A connected arm can already be physically enabled although this
            # process has never run enable()/a detection cycle. Read real limits
            # before returning; do not enable motors or substitute defaults.
            await self.call('prepare_move', home)
            if not self.finishing:
                raise Stopped('收回已中止')
            # Only this explicit button request may resume motion after the
            # cancelled cycle. Disconnect/exit clears finishing and wins here.
            self.stop_requested = False
            opening = True
            await self._release(stream_id)
            self.phase, self.message = 'returning', '本轮已停止，夹爪已张开、接触已解除，正在收回'
            LOGGER.info('停止本轮后开始收回：cycle=%s', self.cycle_id)
            await self._move(home)
            self.phase, self.message = 'complete', '本轮已停止，夹爪已张开并返回收回点；可再次检测'
            LOGGER.info('停止本轮后已收回：cycle=%s', self.cycle_id)
        except Exception as exc:
            self.stop_requested = True
            if isinstance(exc, Stopped):
                self.phase, self.message = 'stopped', '收回已中止，尝试保持当前位置；检查后请断开重连'
                LOGGER.info('停止后的收回被中止：cycle=%s', self.cycle_id)
            else:
                self.phase, self.message = 'fault', f'停止本轮后收回未完成：{exc}；请检查后重连'
                LOGGER.exception('停止本轮后收回失败：cycle=%s', self.cycle_id)
            # Attempt both holds independently; one failed channel must not
            # prevent attempting to stop the other actuator.
            for name in (('hold_gripper', 'hold') if opening else ('hold',)):
                try:
                    await self.call(name)
                except Exception:
                    self.phase = 'fault'
                    actuator = '夹爪' if name == 'hold_gripper' else '机械臂'
                    self.message += f'；{actuator}保持失败，请使用硬件急停'
                    LOGGER.exception('停止收回时保持失败：cycle=%s, actuator=%s', self.cycle_id, name)
        finally:
            self.busy = self.finishing = False

    async def stop(self, owner, *, reason='中止并保持'):
        self.require_owner(owner)
        LOGGER.info('收到停止请求：cycle=%s, phase=%s, reason=%s',
                    self.cycle_id, self.phase, reason)
        self.stop_requested = True
        self.finishing = False
        if self.task and not self.task.done():
            await self.task
        else:
            previous_phase, previous_message = self.phase, self.message
            try:
                await self.call('hold')
            except Exception as exc:
                self.phase, self.message = 'fault', f'停止时保持当前位置失败：{exc}；请使用硬件急停'
                LOGGER.exception('停止时保持失败：cycle=%s', self.cycle_id)
                raise
            else:
                if previous_phase == 'fault':
                    self.phase, self.message = previous_phase, previous_message
                else:
                    self.phase, self.message = 'stopped', '已停止；不自动收回，检查后请断开重连'
            finally:
                self.teaching = False

    async def disconnect(self, owner):
        # Browser disconnect and application shutdown can arrive together.
        async with self._disconnect_lock:
            if self.owner is None and self.arm is None:
                return
            self.require_owner(owner)
            self.busy = True
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
                self.monitor_task = None
            hold_error = None
            try:
                await self.stop(owner, reason='断开机械臂连接')
            except Exception as exc:
                hold_error = exc
                LOGGER.warning('断开时保持当前位置失败：%s；继续关闭通信', exc)
            await self.call('disconnect')
            self.arm, self.owner = None, None
            self.busy = self.teaching = False
            self.stop_requested = False
            self.last_snapshot = None
            self.observation = None
            self.ready_check = None
            self._cycle_stream_id = 0
            self.phase, self.message = 'disconnected', '已断开'
            if hold_error:
                self.message += f'；保持指令失败：{hold_error}'
