import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from robot.nero import NeroArm
from robot.backend import Snapshot
from robot.settings import RobotSettings


class JointLimitTests(unittest.TestCase):
    def setUp(self):
        self.arm = NeroArm(RobotSettings(channel='can_follower'))
        self.arm.robot = Mock()

    @staticmethod
    def reply(joint, lo=-1., hi=1., stamp=None):
        return SimpleNamespace(timestamp=time.time() if stamp is None else stamp,
                               msg=SimpleNamespace(joint_index=joint, min_angle_limit=lo, max_angle_limit=hi))

    def test_lost_first_reply_is_retried_and_all_limits_are_real(self):
        attempts = {}
        def query(joint, **kwargs):
            attempts[joint] = attempts.get(joint, 0) + 1
            self.assertEqual(self.arm._limits, [])
            if joint == 1 and attempts[joint] == 1:
                return None
            return self.reply(joint, -joint/10, joint/10)
        self.arm.robot.get_joint_angle_vel_limits.side_effect = query
        with patch('robot.nero.time.sleep'), self.assertLogs('robot.nero', level='WARNING'):
            self.arm._load_joint_limits()
        self.assertEqual(self.arm._limits, [(-j/10, j/10) for j in range(1, 8)])
        self.assertEqual(self.arm.robot.get_joint_angle_vel_limits.call_args_list,
                         [call(j, timeout=1., min_interval=0.) for j in [1, 1, 2, 3, 4, 5, 6, 7]])
        self.arm.robot.enable.assert_not_called()
        self.arm.robot.move_j.assert_not_called()
        self.arm.robot.set_joint_angle_vel_limits.assert_not_called()

    def test_permanent_timeout_names_joint_and_discards_partial_and_previous_limits(self):
        self.arm._limits = [(-2., 2.)]*7
        self.arm.robot.get_joint_angle_vel_limits.side_effect = lambda joint, **kwargs: (
            None if joint == 3 else self.reply(joint))
        with patch('robot.nero.time.sleep'), self.assertLogs('robot.nero', level='WARNING'), \
                self.assertRaisesRegex(TimeoutError, 'J3.*3 次'):
            self.arm._load_joint_limits()
        self.assertEqual(self.arm._limits, [])
        self.assertEqual(self.arm.robot.get_joint_angle_vel_limits.call_count, 5)
        with self.assertRaises(ValueError):
            self.arm.move([0.]*7)
        self.arm.robot.move_j.assert_not_called()

    def test_malformed_limits_are_not_retried_or_used(self):
        cases = [self.reply(2), self.reply(1, lo=None), self.reply(1, lo=float('nan')),
                 self.reply(1, hi=float('inf')), self.reply(1, lo=2.), self.reply(1, lo=1.),
                 self.reply(1, stamp=float('nan')), SimpleNamespace(msg=SimpleNamespace())]
        for reply in cases:
            with self.subTest(reply=reply):
                self.arm.robot.reset_mock()
                self.arm.robot.get_joint_angle_vel_limits.return_value = reply
                with self.assertRaisesRegex(RuntimeError, 'J1.*无效'):
                    self.arm._load_joint_limits()
                self.assertEqual(self.arm._limits, [])
                self.arm.robot.get_joint_angle_vel_limits.assert_called_once()

    def test_transport_error_is_not_hidden_by_retries(self):
        self.arm.robot.get_joint_angle_vel_limits.side_effect = RuntimeError('CAN disconnected')
        with self.assertRaisesRegex(RuntimeError, 'CAN disconnected'):
            self.arm._load_joint_limits()
        self.arm.robot.get_joint_angle_vel_limits.assert_called_once()
        self.assertEqual(self.arm._limits, [])

    def test_limit_failure_in_enable_prevents_speed_and_mode_commands(self):
        self.arm._check_status = Mock()
        self.arm.robot.enable.return_value = True
        self.arm.robot.get_joint_angle_vel_limits.return_value = None
        with patch('robot.nero.time.sleep'), self.assertLogs('robot.nero', level='WARNING'), \
                self.assertRaises(TimeoutError):
            self.arm.enable()
        self.arm.robot.set_speed_percent.assert_not_called()
        self.arm.robot.set_leader_mode.assert_not_called()
        self.arm.robot.move_j.assert_not_called()
        self.assertEqual(self.arm._limits, [])

    def test_prepare_return_on_fresh_connection_reads_limits_without_enabling(self):
        self.arm.snapshot = Mock(return_value=Snapshot((0.,)*7, .06, False, time.monotonic()))
        self.arm.robot.get_joint_angle_vel_limits.side_effect = lambda joint, **kwargs: self.reply(joint)
        self.assertEqual(self.arm._limits, [])
        self.arm.prepare_move([.1]*7)
        self.assertEqual(self.arm._limits, [(-1., 1.)]*7)
        self.assertEqual(self.arm.robot.get_joint_angle_vel_limits.call_count, 7)
        self.arm.prepare_move([0.]*7)
        self.assertEqual(self.arm.robot.get_joint_angle_vel_limits.call_count, 7)
        self.assertFalse(self.arm._enabled)
        self.assertEqual({c[0] for c in self.arm.robot.mock_calls}, {'get_joint_angle_vel_limits'})

    def test_prepare_return_rejects_out_of_limits_without_moving(self):
        self.arm.snapshot = Mock(return_value=Snapshot((0.,)*7, .06, False, time.monotonic()))
        self.arm.robot.get_joint_angle_vel_limits.side_effect = lambda joint, **kwargs: self.reply(joint)
        with self.assertRaisesRegex(ValueError, '目标超出设备关节限位'):
            self.arm.prepare_move([2.]*7)
        self.assertEqual({c[0] for c in self.arm.robot.mock_calls}, {'get_joint_angle_vel_limits'})

    def test_prepare_return_never_reads_limits_through_bad_feedback(self):
        self.arm.snapshot = Mock(side_effect=RuntimeError('反馈异常'))
        with self.assertRaisesRegex(RuntimeError, '反馈异常'):
            self.arm.prepare_move([0.]*7)
        self.assertEqual(self.arm.robot.mock_calls, [])
