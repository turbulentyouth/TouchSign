"""NERO firmware-native drag teaching. No host-side MIT torque loop."""
import time
import math
from robot.backend import Snapshot
from robot.settings import joints


class NeroArm:
    def __init__(self, settings):
        self.settings = settings
        self.robot = None
        self.teaching = False
        self.version = ''
        self.profile = ''
        self._stamps = {}
        self._limits = []

    def connect(self):
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        from pyAgxArm.api.firmware import resolve_firmware_profile
        def create(profile):
            return AgxArmFactory.create_arm(create_agx_arm_config(
                robot='nero', firmeware_version=profile,
                interface='agx_cando', channel=self.settings.channel))
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
            self.profile = f'nero:{self.settings.channel}:{self.version}'
            # Connecting never enables motors or switches mode.
        except Exception:
            self.robot.disconnect()
            self.robot = None
            raise

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

    def snapshot(self):
        if self.robot is None:
            raise RuntimeError('机械臂未连接')
        if self.teaching:
            q = self._fresh('leader joints', self.robot.get_leader_joint_angles())
            # Leader mode may stop ordinary arm/gripper telemetry on older firmware.
            return Snapshot(joints(q), self.settings.open_width, True, time.monotonic(), self.version)
        q = self._fresh('joints', self.robot.get_joint_angles())
        status = self._fresh('status', self.robot.get_arm_status())
        if status.arm_status != 0:
            raise RuntimeError(f'NERO 故障状态：{status.arm_status}')
        g = self._fresh('gripper', self.effector.get_gripper_status())
        if g.mode != 'width':
            raise RuntimeError('需要夹爪宽度模式反馈')
        for flag in ('voltage_too_low', 'motor_overheating', 'driver_overcurrent',
                     'driver_overheating', 'driver_error_status'):
            if getattr(g.foc_status, flag, False):
                raise RuntimeError(f'夹爪故障：{flag}')
        return Snapshot(joints(q), g.value, False, time.monotonic(), self.version)

    def enable(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.robot.enable():
                self.robot.set_speed_percent(self.settings.speed_percent)
                self._limits = []
                for i in range(1, 8):
                    msg = self.robot.get_joint_angle_vel_limits(i, timeout=.5)
                    if msg is None:
                        raise RuntimeError('无法读取关节限位，禁止运动')
                    self._limits.append((msg.msg.min_angle_limit, msg.msg.max_angle_limit))
                return
            time.sleep(.05)
        raise TimeoutError('机械臂使能超时')

    def _position_mode(self):
        major, minor = [int(x) for x in self.version.split('.')[:2]]
        if (major, minor) < (1, 12):
            self.robot.set_normal_mode()
        else:
            self.robot.set_follower_mode()

    def enter_teach(self):
        self.snapshot()
        self.robot.set_leader_mode()
        self.teaching = True
        self._stamps.clear()

    def exit_teach(self):
        q = self.snapshot().q
        self._position_mode()
        self.robot.set_speed_percent(self.settings.speed_percent)
        self.robot.move_j(list(q))
        self.teaching = False
        self._stamps.clear()

    def move(self, q):
        if self.teaching:
            raise RuntimeError('必须先退出重力补偿示教')
        q = joints(q)
        if len(self._limits) != 7 or any(not lo <= x <= hi for x, (lo, hi) in zip(q, self._limits)):
            raise ValueError('目标超出设备关节限位，或限位未读取')
        self.snapshot()
        self.robot.set_speed_percent(self.settings.speed_percent)
        self.robot.move_j(list(q))

    def gripper(self, width):
        if self.teaching:
            raise RuntimeError('示教中禁止夹爪运动')
        self.effector.move_gripper_m(value=width, force=self.settings.gripper_force)

    def hold(self):
        # Best-effort position hold using fresh feedback, never reset/disable.
        if self.teaching:
            self.exit_teach()
        else:
            self.robot.move_j(list(self.snapshot().q))

    def disconnect(self):
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
