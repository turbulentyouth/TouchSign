"""A deterministic offline arm for exercising the same control sequence."""
from dataclasses import dataclass
import time
from robot.settings import joints


class ArmNeedsEnable(RuntimeError):
    """Initial brake-closed state, before this connection has enabled the arm."""


@dataclass(frozen=True)
class Snapshot:
    q: tuple
    width: float
    teaching: bool
    timestamp: float
    firmware: str = 'SIMULATED'


class MockArm:
    profile = 'mock'
    def __init__(self, settings):
        self.settings = settings
        self.q = (0.,) * 7
        self.width = settings.open_width
        self.teaching = False
        self.connected = False
        self.events = []
        self._target = self.q
        self._arrival = 0.
        self.fail_feedback = False
        self.stuck_gripper = False

    def connect(self):
        self.connected = True

    def snapshot(self):
        if not self.connected or self.fail_feedback:
            raise RuntimeError('模拟通信中断')
        if time.monotonic() >= self._arrival:
            self.q = self._target
        return Snapshot(self.q, self.width, self.teaching, time.monotonic())

    def enable(self):
        self.snapshot()

    def enter_teach(self):
        self.snapshot()
        self.teaching = True
        self.events.append('teach')

    def exit_teach(self):
        q = self.snapshot().q
        self.teaching = False
        self._target = q
        self.events.append('hold')

    def simulate_drag(self, q):
        if not self.teaching:
            raise RuntimeError('请先开始示教')
        self.q = self._target = joints(q)

    def prepare_move(self, q):
        if self.teaching:
            raise RuntimeError('示教中不能自动运动')
        self.snapshot()
        joints(q)

    def move(self, q):
        if self.teaching:
            raise RuntimeError('示教中不能自动运动')
        self.snapshot()
        self._target = joints(q)
        self._arrival = time.monotonic() + .3
        self.events.append(('move', self._target))

    def gripper(self, width):
        self.snapshot()
        if not self.stuck_gripper:
            self.width = width
        self.events.append(('gripper', width))

    def hold(self):
        self._target = self.q
        self.teaching = False
        self.events.append('hold')

    def hold_gripper(self):
        width = self.snapshot().width
        self.events.append(('hold_gripper', width))
        return width

    def disconnect(self):
        self.connected = False
