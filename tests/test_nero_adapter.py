import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robot.nero import NeroArm
from robot.settings import RobotSettings
from robot.backend import ArmNeedsEnable, Snapshot


class NeroAdapterTests(unittest.TestCase):
    def test_initial_feedback_waits_for_receiver_without_enabling(self):
        arm = self.braked_arm()
        arm.robot.get_arm_status.side_effect = [None, None,
            SimpleNamespace(msg=SimpleNamespace(arm_status=6))]
        with patch('robot.nero.time.sleep') as sleep:
            arm._wait_initial_feedback()
        self.assertEqual(sleep.call_count, 2)
        arm.robot.enable.assert_not_called()
        arm.robot.move_j.assert_not_called()

    def test_initial_feedback_wait_has_a_deadline(self):
        arm = self.braked_arm()
        arm.robot.get_arm_status.side_effect = None
        arm.robot.get_arm_status.return_value = None
        with patch('robot.nero.time.monotonic', side_effect=[0., 0., 3.]), \
                patch('robot.nero.time.sleep'):
            with self.assertRaisesRegex(TimeoutError, '初始反馈'):
                arm._wait_initial_feedback()
        arm.robot.enable.assert_not_called()

    def braked_arm(self, code=6, err_code=0):
        arm = NeroArm(RobotSettings())
        arm.robot = Mock()
        arm.effector = Mock()
        arm.robot.get_arm_status.side_effect = lambda: SimpleNamespace(
            timestamp=time.monotonic(), msg=SimpleNamespace(arm_status=code, err_code=err_code))
        return arm

    def test_initial_brake_is_waiting_enable_and_hold_sends_no_motion(self):
        arm = self.braked_arm()
        with self.assertRaises(ArmNeedsEnable):
            arm.snapshot()
        arm.hold()
        arm.robot.enable.assert_not_called()
        arm.robot.move_j.assert_not_called()
        arm.robot.set_leader_mode.assert_not_called()

    def test_brake_after_enable_and_other_faults_remain_faults(self):
        for code, error, enabled in ((6, 0, True), (1, 0, False), (6, 1, False)):
            with self.subTest(code=code, error=error, enabled=enabled):
                arm = self.braked_arm(code, error)
                arm._enabled = enabled
                with self.assertRaises(RuntimeError) as caught:
                    arm.enable()
                self.assertNotIsInstance(caught.exception, ArmNeedsEnable)
                arm.robot.enable.assert_not_called()

    def test_enable_requires_brake_release_even_after_sdk_success(self):
        arm = self.braked_arm()
        clock = SimpleNamespace(now=0.)
        def sleep(seconds):
            clock.now += seconds
        with patch('robot.nero.time') as fake_time:
            fake_time.monotonic.side_effect = lambda: clock.now
            fake_time.sleep.side_effect = sleep
            with self.assertRaisesRegex(TimeoutError, '抱闸仍未释放'):
                arm.enable()
        arm.robot.set_speed_percent.assert_not_called()
        arm.robot.set_leader_mode.assert_not_called()
        arm.robot.move_j.assert_not_called()
        self.assertFalse(arm._enabled)

    def test_enable_waits_for_normal_status_then_loads_limits(self):
        arm = self.braked_arm()
        codes = iter((6, 6, 0))
        arm.robot.get_arm_status.side_effect = lambda: SimpleNamespace(
            timestamp=time.monotonic(), msg=SimpleNamespace(arm_status=next(codes), err_code=0))
        arm.robot.get_joint_angle_vel_limits.side_effect = lambda joint, **kwargs: SimpleNamespace(
            timestamp=time.time(), msg=SimpleNamespace(joint_index=joint, min_angle_limit=-1., max_angle_limit=1.))
        arm.enable()
        self.assertTrue(arm._enabled)
        self.assertEqual(arm._limits, [(-1., 1.)]*7)
        arm.robot.move_j.assert_not_called()
        arm.robot.set_leader_mode.assert_not_called()

    def test_exit_teach_holds_current_pose_for_each_firmware(self):
        for version, mode in [('1.11', 'set_normal_mode'), ('1.12', 'set_follower_mode'), ('1.21', 'set_follower_mode')]:
            with self.subTest(version=version):
                arm = NeroArm(RobotSettings())
                arm.robot = Mock()
                arm.version = version
                arm.teaching = True
                arm.snapshot = Mock(return_value=Snapshot((.1,)*7, .06, True, time.monotonic(), version))
                arm._wait_mode_feedback = Mock()
                arm.effector = Mock()
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

    def test_gripper_rejects_targets_outside_configured_opening(self):
        arm = NeroArm(RobotSettings(close_width=.005))
        arm.effector = Mock()
        arm.snapshot = Mock(return_value=Snapshot((0.,)*7, .01, False, time.monotonic()))
        for width in (0., .0049, .061, float('nan'), float('inf')):
            with self.subTest(width=width), self.assertRaises(ValueError):
                arm.gripper(width)
        arm.effector.move_gripper_m.assert_not_called()
        arm.gripper(.005)
        arm.effector.move_gripper_m.assert_called_once_with(value=.005, force=1.)

    def test_gripper_hold_uses_actual_width_and_never_resets_or_disables(self):
        arm = NeroArm(RobotSettings(close_width=.005))
        arm.effector = Mock()
        arm.snapshot = Mock(return_value=Snapshot((0.,)*7, .0096, False, time.monotonic()))
        self.assertEqual(arm.hold_gripper(), .0096)
        arm.effector.move_gripper_m.assert_called_once_with(value=.0096, force=1.)
        self.assertEqual(len(arm.effector.mock_calls), 1)
        arm.effector.reset_mock()
        arm.snapshot.side_effect = RuntimeError('stale')
        for action in (lambda: arm.gripper(.009), arm.hold_gripper):
            with self.assertRaises(RuntimeError):
                action()
        arm.effector.move_gripper_m.assert_not_called()

    def test_invalid_gripper_width_feedback_is_rejected(self):
        for width in (float('nan'), float('inf'), -.001, .101):
            arm = NeroArm(RobotSettings())
            arm.robot = Mock()
            arm.effector = Mock()
            arm._check_status = Mock()
            arm._fresh = Mock(side_effect=[(0.,)*7,
                SimpleNamespace(mode='width', value=width, foc_status=SimpleNamespace())])
            with self.subTest(width=width), self.assertRaisesRegex(RuntimeError, '开口反馈无效'):
                arm.snapshot()


if __name__ == '__main__':
    unittest.main()
