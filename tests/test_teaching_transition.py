"""Delayed/stalled CAN feedback must not falsely confirm a mode transition."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robot.nero import NeroArm
from robot.settings import RobotSettings


class TeachingTransitionTests(unittest.TestCase):
    def setUp(self):
        self.clock = SimpleNamespace(now=0.)
        self.mode = 'position'
        self.ready_at = 0.
        self.can_push = True
        self.arm = NeroArm(RobotSettings())
        self.arm.version = '1.22'
        self.arm._enabled = True
        self.arm.robot = Mock()
        self.arm.robot._msg_mode = SimpleNamespace(
            move_mode=1, enable_can_push=0,
            Enums=SimpleNamespace(CanActiveMsgReporting=SimpleNamespace(ENABLE=1)))
        self.arm.robot._send_msg.side_effect = self.send
        self.arm.effector = Mock()
        self.arm.robot.get_arm_status.side_effect = lambda: self.packet(SimpleNamespace(
            arm_status=0, err_code=0, ctrl_mode=1, teach_status=0))
        self.arm.robot.get_joint_angles.side_effect = lambda: self.packet([.2]*7)
        self.arm.effector.get_gripper_status.side_effect = lambda: self.packet(SimpleNamespace(
            mode='width', value=.06, foc_status=SimpleNamespace()))
        self.arm.robot.get_leader_joint_angles.side_effect = self.leader
        self.arm.robot.set_leader_mode.side_effect = lambda: self.switch('leader')
        self.arm.robot.set_follower_mode.side_effect = lambda: self.switch('position')
        self.time_patch = patch('robot.nero.time')
        timer = self.time_patch.start()
        self.addCleanup(self.time_patch.stop)
        timer.monotonic.side_effect = lambda: self.clock.now
        timer.sleep.side_effect = self.sleep

    def sleep(self, seconds):
        self.clock.now += seconds

    def switch(self, mode):
        self.mode = mode
        self.ready_at = self.clock.now + .12
        # The real SDK disables ordinary CAN push for BOTH leader and follower.
        self.can_push = False

    def send(self, message):
        if message.enable_can_push == 1:
            self.can_push = True

    def packet(self, value):
        # Ordinary telemetry remains cached while leader mode is active and
        # during the subsequent switch back to position control.
        stamp = self.clock.now if (self.can_push and self.mode == 'position'
                                  and self.clock.now >= self.ready_at) else 0.
        return SimpleNamespace(timestamp=stamp, msg=value)

    def leader(self):
        if self.mode != 'leader' or self.clock.now < self.ready_at:
            return None
        return SimpleNamespace(timestamp=self.clock.now, msg=[.2]*7)

    def test_delayed_first_leader_frames_confirm_without_repeating_commands(self):
        self.arm.enter_teach()
        self.assertGreaterEqual(self.clock.now, .14)
        self.assertTrue(self.arm.snapshot().teaching)
        self.arm.robot.set_leader_mode.assert_called_once()
        self.arm.robot.move_j.assert_not_called()
        self.arm.robot.disable.assert_not_called()

    def test_missing_leader_never_falls_back_to_ordinary_joints(self):
        self.arm.robot.get_leader_joint_angles.side_effect = lambda: None
        with self.assertRaisesRegex(TimeoutError, 'leader joints.*模式未确认'):
            self.arm.enter_teach()
        self.assertGreaterEqual(self.clock.now, 3.)
        self.assertTrue(self.arm.teaching)  # mode may have changed physically
        self.arm.robot.set_leader_mode.assert_called_once()
        self.arm.robot.set_follower_mode.assert_not_called()
        self.arm.robot.move_j.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'leader joints'):
            self.arm.hold()
        self.arm.robot.move_j.assert_not_called()

    def test_old_teaching_cache_is_not_a_new_mode_confirmation(self):
        old = SimpleNamespace(timestamp=10., msg=[.3]*7)
        self.arm.robot.get_leader_joint_angles.side_effect = lambda: old
        with self.assertRaisesRegex(TimeoutError, 'leader joints'):
            self.arm.enter_teach()
        with self.assertRaisesRegex(RuntimeError, '未更新'):
            self.arm.hold()
        self.arm.robot.set_follower_mode.assert_not_called()
        self.arm.robot.move_j.assert_not_called()

    def test_single_leader_packet_does_not_prove_feedback_is_running(self):
        self.arm.robot.get_leader_joint_angles.side_effect = lambda: (
            SimpleNamespace(timestamp=10., msg=[.3]*7) if self.mode == 'leader' else None)
        with self.assertRaises(TimeoutError):
            self.arm.enter_teach()

    def test_mutable_sdk_message_still_detects_timestamp_updates(self):
        packet = SimpleNamespace(timestamp=0., msg=[.2]*7)
        def read():
            if self.mode == 'leader' and self.clock.now >= self.ready_at:
                packet.timestamp = self.clock.now
            return packet
        self.arm.robot.get_leader_joint_angles.side_effect = read
        self.arm.enter_teach()
        self.assertTrue(self.arm.snapshot().teaching)

    def test_invalid_leader_values_are_not_retried_or_used_for_motion(self):
        for stamp, q in ((float('nan'), [.2]*7), (1., [.2]*6)):
            with self.subTest(stamp=stamp):
                self.arm.teaching = False
                self.mode = 'position'
                self.can_push = True
                self.ready_at = 0.
                self.arm.robot.get_leader_joint_angles.side_effect = lambda: (
                    SimpleNamespace(timestamp=stamp, msg=q) if self.mode == 'leader' else None)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.arm.enter_teach()
                self.arm.robot.move_j.assert_not_called()

    def test_running_teach_still_rejects_stale_feedback(self):
        self.arm.enter_teach()
        packet = self.arm.robot.get_leader_joint_angles()
        self.arm.robot.get_leader_joint_angles.side_effect = lambda: packet
        self.sleep(.6)
        with self.assertRaisesRegex(RuntimeError, '未更新'):
            self.arm.snapshot()

    def test_exit_waits_for_new_position_feedback_and_holds_current_pose(self):
        self.arm.enter_teach()
        started = self.clock.now
        self.arm.exit_teach()
        self.assertGreaterEqual(self.clock.now - started, .14)
        self.assertFalse(self.arm.snapshot().teaching)
        self.arm.robot.set_follower_mode.assert_called_once()
        self.arm.robot.move_j.assert_called_once_with([.2]*7)
        self.arm.robot.disable.assert_not_called()

    def test_exit_without_restored_push_rejects_normal_but_cached_status(self):
        self.arm.enter_teach()
        self.arm.robot._send_msg.side_effect = lambda message: None
        with self.assertRaisesRegex(TimeoutError, 'status, joints, gripper'):
            self.arm.exit_teach()
        self.arm.robot.move_j.assert_called_once_with([.2]*7)
        with self.assertRaisesRegex(RuntimeError, '未更新'):
            self.arm.hold()
        self.arm.robot.move_j.assert_called_once()
