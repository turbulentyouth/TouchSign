import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from robot.nero import NeroArm
from robot.settings import RobotSettings
from robot.backend import Snapshot


class NeroAdapterTests(unittest.TestCase):
    def test_exit_teach_holds_current_pose_for_each_firmware(self):
        for version, mode in [('1.11', 'set_normal_mode'), ('1.12', 'set_follower_mode'), ('1.21', 'set_follower_mode')]:
            with self.subTest(version=version):
                arm = NeroArm(RobotSettings())
                arm.robot = Mock()
                arm.version = version
                arm.teaching = True
                arm.snapshot = Mock(return_value=Snapshot((.1,)*7, .06, True, time.monotonic(), version))
                arm.exit_teach()
                getattr(arm.robot, mode).assert_called_once_with()
                arm.robot.move_j.assert_called_once_with([.1]*7)
                arm.robot.disable.assert_not_called()
                self.assertFalse(arm.teaching)

    def test_limits_required_and_out_of_range_never_sent(self):
        arm = NeroArm(RobotSettings())
        arm.robot = Mock()
        with self.assertRaises(ValueError):
            arm.move([0]*7)
        arm._limits = [(-.5, .5)]*7
        with self.assertRaises(ValueError):
            arm.move([1]*7)
        arm.robot.move_j.assert_not_called()

    def test_malformed_and_stale_feedback_rejected(self):
        arm = NeroArm(RobotSettings())
        for stamp in (None, float('nan'), float('inf')):
            with self.assertRaises(RuntimeError):
                arm._fresh('q', SimpleNamespace(timestamp=stamp, msg=[0]*7))
        arm._stamps['q'] = (1, time.monotonic()-1)
        with self.assertRaises(RuntimeError):
            arm._fresh('q', SimpleNamespace(timestamp=1, msg=[0]*7))

    def test_hold_does_not_use_stale_pose_or_disable(self):
        arm = NeroArm(RobotSettings())
        arm.robot = Mock()
        arm.snapshot = Mock(side_effect=RuntimeError('stale'))
        with self.assertRaises(RuntimeError):
            arm.hold()
        arm.robot.move_j.assert_not_called()
        arm.robot.disable.assert_not_called()


if __name__ == '__main__':
    unittest.main()
