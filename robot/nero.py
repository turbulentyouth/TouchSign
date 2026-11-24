"""NERO firmware-native drag teaching. No host-side MIT torque loop."""
import time
import math
import logging
from copy import copy
from robot.backend import ArmNeedsEnable, Snapshot
from robot.settings import joints
from robot.transport import can_interface, check_socketcan

LOGGER = logging.getLogger(__name__)


class NeroArm:
    def __init__(self, settings):
        self.settings = settings
        self.robot = None
        self.teaching = False
        self.version = ''
        self.profile = ''
        self._stamps = {}
        self._limits = []
        self._enabled = False

    def connect(self):
        self._enabled = False
        self._limits = []
        self._stamps.clear()
        interface = can_interface()
        if interface == 'socketcan':
            check_socketcan(self.settings.channel)
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        from pyAgxArm.api.firmware import resolve_firmware_profile
        def create(profile):
            return AgxArmFactory.create_arm(create_agx_arm_config(
                robot='nero', firmeware_version=profile,
                interface=interface, channel=self.settings.channel,
                auto_connect=False))
        probe = create('default')
        try:
            probe.connect()
            info = probe.get_firmware(timeout=2.0)
            if not info:
                raise RuntimeError('无法读取 NERO 固件；检查 USB-CAN、供电与驱动')
            self.version = info['software_version']
        finally:
            probe.disconnect()
        self.robot = create(resolve_firmware_profile('nero', self.version))
        try:
            self.robot.connect()
            self.effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            self._wait_initial_feedback()
            self.profile = f'nero:{self.settings.channel}:{self.version}'
            # Connecting never enables motors or switches mode.
        except Exception:
            self.robot.disconnect()
            self.robot = None
            raise

    def _wait_initial_feedback(self):
        # The SDK starts its receiver asynchronously; connect() can return
        # before the first CAN frame. This grace period applies only at startup.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = self.robot.get_arm_status()
            if status is not None:
                # Braked/faulted arms need only status to be classified safely.
                if status.msg.arm_status != 0:
                    return
                if (self.robot.get_joint_angles() is not None
                        and self.effector.get_gripper_status() is not None):
                    return
            time.sleep(.02)
        raise TimeoutError('连接后 2 秒内未收到机械臂初始反馈；检查 CAN 和设备供电')

    def _fresh(self, name, message):
        if message is None:
            raise RuntimeError(f'缺少 {name} 反馈')
        stamp = message.timestamp
        if stamp is None or not math.isfinite(float(stamp)):
            raise RuntimeError(f'{name} 时间戳无效')
        now = time.monotonic()
        old_stamp, received = self._stamps.get(name, (None, now))
        if stamp != old_stamp:
            self._stamps[name] = (stamp, now)
        elif now - received > .5:
            raise RuntimeError(f'{name} 反馈超过 0.5 秒未更新')
        return message.msg

    def _check_status(self):
        status = self._fresh('status', self.robot.get_arm_status())
        if getattr(status, 'err_code', 0):
            raise RuntimeError(f'NERO 关节故障码：{status.err_code:#06x}')
        if status.arm_status == 6 and not self._enabled:
            raise ArmNeedsEnable('NERO 已连接；抱闸未释放，等待点击“开始重力补偿示教”使能')
        if status.arm_status != 0:
            raise RuntimeError(f'NERO 故障状态：{status.arm_status}')

    def snapshot(self):
        if self.robot is None:
            raise RuntimeError('机械臂未连接')
        if self.teaching:
            q = self._fresh('leader joints', self.robot.get_leader_joint_angles())
            # Leader mode may stop ordinary arm/gripper telemetry on older firmware.
            return Snapshot(joints(q), self.settings.open_width, True, time.monotonic(), self.version)
        self._check_status()
        q = self._fresh('joints', self.robot.get_joint_angles())
        g = self._fresh('gripper', self.effector.get_gripper_status())
        if g.mode != 'width':
            raise RuntimeError('需要夹爪宽度模式反馈')
        if not math.isfinite(g.value) or not 0 <= g.value <= .1:
            raise RuntimeError('夹爪开口反馈无效')
        for flag in ('voltage_too_low', 'motor_overheating', 'driver_overcurrent',
                     'driver_overheating', 'driver_error_status'):
            if getattr(g.foc_status, flag, False):
                raise RuntimeError(f'夹爪故障：{flag}')
        return Snapshot(joints(q), g.value, False, time.monotonic(), self.version)

    def enable(self):
        # Only an explicit action reaches here. Do not enable through other faults.
        self._limits = []
        try:
            self._check_status()
        except ArmNeedsEnable:
            pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.robot.enable():
                # Motor-enable acknowledgements alone do not prove brake release.
                while time.monotonic() < deadline:
                    try:
                        self._check_status()
                    except ArmNeedsEnable:
                        time.sleep(.05)
                        continue
                    self._enabled = True
                    break
                else:
                    raise TimeoutError('机械臂使能超时：抱闸仍未释放，未进入示教')
                self._load_joint_limits()
                self.robot.set_speed_percent(self.settings.speed_percent)
                return
            time.sleep(.05)
        raise TimeoutError('机械臂使能超时')

    def _load_joint_limits(self):
        """Read all seven limits, retry lost replies, and publish atomically."""
        self._limits = []
        limits = []
        for joint in range(1, 8):
            for attempt in range(1, 4):
                # We bound retries here. Disable the SDK's request throttle so
                # each retry really sends a query rather than only waiting again.
                message = self.robot.get_joint_angle_vel_limits(
                    joint, timeout=1., min_interval=0.)
                if message is not None:
                    break
                LOGGER.warning('关节 J%d 限位查询第 %d/3 次超时，channel=%s',
                               joint, attempt, self.settings.channel)
                if attempt < 3:
                    time.sleep(.05)
            else:
                raise TimeoutError(f'无法读取关节 J{joint} 限位：已查询 3 次，每次等待 1 秒；'
                                   '禁止运动，请检查 CAN 通信后重连')
            try:
                lo, hi = float(message.msg.min_angle_limit), float(message.msg.max_angle_limit)
                valid = (message.msg.joint_index == joint and math.isfinite(lo)
                         and math.isfinite(hi) and lo < hi
                         and math.isfinite(float(message.timestamp)))
            except (AttributeError, TypeError, ValueError):
                valid = False
            if not valid:
                raise RuntimeError(f'关节 J{joint} 限位回复无效（关节编号、角度范围或时间戳异常）；禁止运动')
            limits.append((lo, hi))
        self._limits = limits
        LOGGER.info('已读取并校验七关节限位，channel=%s，limits_rad=%s',
                    self.settings.channel, limits)

    def _position_mode(self):
        major, minor = [int(x) for x in self.version.split('.')[:2]]
        if (major, minor) < (1, 12):
            self.robot.set_normal_mode()
        else:
            self.robot.set_follower_mode()
            self._resume_can_feedback()

    def _resume_can_feedback(self):
        # The pinned SDK disables CAN push in set_follower_mode(), including
        # v112/v120/v121, where set_normal_mode() is a no-op. Reuse its mode
        # message/encoder, keeping existing settings and the SDK's "no change"
        # move_mode=0xFF. This belongs to the explicit exit-teach operation;
        # it is not a read-only command or a reason to enable motors again.
        # Send a copy so a failed write cannot poison subsequent SDK commands.
        mode = copy(self.robot._msg_mode)
        mode.move_mode = 0xFF
        mode.enable_can_push = mode.Enums.CanActiveMsgReporting.ENABLE
        self.robot._send_msg(mode)
        LOGGER.info('退出示教：已请求恢复普通 CAN 反馈，等待新数据确认，channel=%s',
                    self.settings.channel)

    def _mode_readers(self, teaching):
        if teaching:
            return {'leader joints': self.robot.get_leader_joint_angles}
        return {'status': self.robot.get_arm_status,
                'joints': self.robot.get_joint_angles,
                'gripper': self.effector.get_gripper_status}

    @staticmethod
    def _feedback_stamps(readers):
        # SDK message objects are mutable: retain scalar timestamps, not objects.
        result = {}
        for name, read in readers.items():
            message = read()
            result[name] = None if message is None else message.timestamp
        return result

    def _wait_mode_feedback(self, readers, previous, timeout=3.):
        """Confirm new, updating telemetry after a single mode command."""
        deadline = time.monotonic() + timeout
        updates = dict.fromkeys(readers, 0)
        for name, stamp in previous.items():
            if stamp is not None:
                # Preserve the age of pre-switch caches, including on timeout:
                # a later stop/hold must never treat them as newly received.
                self._stamps[name] = (stamp, time.monotonic())
        while time.monotonic() < deadline:
            for name, read in readers.items():
                message = read()
                if message is None:
                    continue  # A mode transition can briefly stop the CAN stream.
                stamp = message.timestamp
                if stamp is None or not math.isfinite(float(stamp)):
                    raise RuntimeError(f'{name} 时间戳无效')
                if stamp != previous[name]:
                    value = self._fresh(name, message)
                    if name in ('joints', 'leader joints'):
                        joints(value)
                    elif name == 'status':
                        self._check_status()
                    previous[name] = stamp
                    updates[name] += 1
            # One cached packet (including one from a previous teach session)
            # cannot prove the mode changed or that feedback is still alive.
            if all(count >= 2 for count in updates.values()):
                return self.snapshot()
            time.sleep(.02)
        missing = ', '.join(name for name, count in updates.items() if count < 2)
        target = '示教' if self.teaching else '位置保持'
        status = self.robot.get_arm_status()
        detail = '无普通状态反馈'
        if status is not None:
            s = status.msg
            detail = (f'状态缓存：ctrl_mode={getattr(s, "ctrl_mode", "未知")}, '
                      f'arm_status={s.arm_status}, teach_status={getattr(s, "teach_status", "未知")}')
        raise TimeoutError(f'切换{target}超时：{timeout:g} 秒内未收到持续更新的 {missing}；'
                           f'{detail}；模式未确认，请扶稳机械臂后停止并断开重连')

    def enter_teach(self):
        self.snapshot()
        readers = self._mode_readers(True)
        before = self._feedback_stamps(readers)
        self.robot.set_leader_mode()
        # The arm may already be draggable even while feedback is pending.
        # Keep this flag on failure so hold() never assumes ordinary control.
        self.teaching = True
        self._stamps.clear()
        self._wait_mode_feedback(readers, before)

    def exit_teach(self):
        q = self.snapshot().q
        readers = self._mode_readers(False)
        before = self._feedback_stamps(readers)
        self._position_mode()
        self.robot.set_speed_percent(self.settings.speed_percent)
        self.robot.move_j(list(q))
        self.teaching = False
        self._stamps.clear()
        self._wait_mode_feedback(readers, before)

    def _validate_move_target(self, q):
        if self.teaching:
            raise RuntimeError('必须先退出重力补偿示教')
        q = joints(q)
        if len(self._limits) != 7:
            raise ValueError('设备关节限位未读取，禁止运动')
        if any(not lo <= x <= hi for x, (lo, hi) in zip(q, self._limits)):
            raise ValueError('目标超出设备关节限位，禁止运动')
        return q

    def prepare_move(self, q):
        """Read/validate limits for an explicit return without enabling motors."""
        if self.teaching:
            raise RuntimeError('必须先退出重力补偿示教')
        q = joints(q)
        self.snapshot()
        if len(self._limits) != 7:
            self._load_joint_limits()
        self._validate_move_target(q)

    def move(self, q):
        q = self._validate_move_target(q)
        self.snapshot()
        self.robot.set_speed_percent(self.settings.speed_percent)
        self.robot.move_j(list(q))

    def gripper(self, width):
        if self.teaching:
            raise RuntimeError('示教中禁止夹爪运动')
        if not math.isfinite(width) or not self.settings.close_width <= width <= self.settings.open_width:
            raise ValueError('夹爪目标超出配置的闭合目标间距或张开宽度')
        self.snapshot()
        self.effector.move_gripper_m(value=width, force=self.settings.gripper_force)

    def hold_gripper(self):
        if self.teaching:
            raise RuntimeError('示教中禁止夹爪运动')
        # Cancel an outstanding closing target with fresh actual width. This is
        # position hold, not a torque disable, reset, or zero calibration.
        width = self.snapshot().width
        self.effector.move_gripper_m(value=width, force=self.settings.gripper_force)
        return width

    def hold(self):
        # Best-effort position hold using fresh feedback, never reset/disable.
        if self.teaching:
            self.exit_teach()
        else:
            try:
                q = self.snapshot().q
            except ArmNeedsEnable:
                # Never command motion to an arm still awaiting its first enable.
                return
            self.robot.move_j(list(q))

    def disconnect(self):
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
            self._enabled = False
            self._limits = []
