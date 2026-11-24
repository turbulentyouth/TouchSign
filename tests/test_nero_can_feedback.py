"""Exercise the installed SDK encoders without opening a CAN connection."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from pyAgxArm import AgxArmFactory, create_agx_arm_config

from robot.backend import Snapshot
from robot.nero import NeroArm
from robot.settings import RobotSettings


class NeroCanFeedbackTests(unittest.TestCase):
    PROFILES = (('default', '1.10'), ('v111', '1.11'), ('v112', '1.12'),
                ('v120', '1.20'), ('v121', '1.22'))

    def adapter(self, profile, version):
        arm = NeroArm(RobotSettings())
        arm.version = version
        arm.robot = AgxArmFactory.create_arm(create_agx_arm_config(
            robot='nero', firmeware_version=profile, interface='socketcan',
            channel='offline-test', auto_connect=False, enable_check_can=False))
        self.addCleanup(arm.robot.disconnect)
        # Fail if these offline tests accidentally try to acquire a CAN device.
        original_get_comm = arm.robot._ctx.get_comm
        arm.robot._ctx.get_comm = Mock(side_effect=AssertionError('must stay offline'))
        self.addCleanup(setattr, arm.robot._ctx, 'get_comm', original_get_comm)
        self.addCleanup(arm.robot._ctx.get_comm.assert_not_called)
        arm.effector = Mock()
        return arm

    def test_exit_restores_push_with_actual_sdk_for_all_firmware_profiles(self):
        for profile, version in self.PROFILES:
            with self.subTest(profile=profile):
                arm = self.adapter(profile, version)
                frames = []
                state = SimpleNamespace(push=True)

                def send(message):
                    frame = arm.robot._parser.pack(message)
                    self.assertIsNotNone(frame)
                    frames.append(frame)
                    if frame.arbitration_id == 0x151 and frame.data[6] in (1, 2):
                        state.push = frame.data[6] == 1

                arm.robot._send_msg = send
                arm.robot.set_leader_mode()
                self.assertFalse(state.push)
                frames.clear()
                arm.teaching = True
                arm.snapshot = Mock(return_value=Snapshot((.1,)*7, .06, True, 1., version))
                arm._wait_mode_feedback = Mock()
                arm.exit_teach()

                self.assertTrue(state.push, 'ordinary feedback must resume after teaching')
                ids = {frame.arbitration_id for frame in frames}
                self.assertEqual(ids, {0x151, 0x470, 0x155, 0x156, 0x157, 0x170})
                # Mode, linkage and the existing current-pose hold only. No motor
                # enable/disable, gripper motion, reset, or limit-write commands.
                link = next(frame for frame in frames if frame.arbitration_id == 0x470)
                self.assertEqual(link.data[0], 0 if profile in ('default', 'v111') else 0xFC)
                enabled = [frame for frame in frames
                           if frame.arbitration_id == 0x151 and frame.data[6] == 1]
                self.assertEqual(len(enabled), 1)
                self.assertEqual(enabled[0].data[1], 0xFF)
                # Subsequent speed/hold commands must not switch push off again.
                enabled_index = frames.index(enabled[0])
                self.assertFalse(any(frame.arbitration_id == 0x151 and frame.data[6] == 2
                                     for frame in frames[enabled_index + 1:]))
                self.assertEqual(arm.robot._msg_mode.enable_can_push, 0)
                self.assertEqual(arm.robot._msg_mode.move_mode, 1)
                self.assertEqual(arm.robot._msg_mode.move_spd_rate_ctrl, arm.settings.speed_percent)
                arm._wait_mode_feedback.assert_called_once()

    def test_feedback_request_preserves_cached_settings_even_on_send_failure(self):
        for fail in (False, True):
            with self.subTest(send_failure=fail):
                arm = self.adapter('v121', '1.22')
                mode = arm.robot._msg_mode
                mode.move_spd_rate_ctrl = 10
                mode.installation_pos = 2
                before = vars(mode).copy()
                frames = []

                def send(message):
                    frames.append(arm.robot._parser.pack(message))
                    if fail:
                        raise RuntimeError('CAN write failed')

                arm.robot._send_msg = send
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'CAN write failed'):
                        arm._resume_can_feedback()
                else:
                    arm._resume_can_feedback()
                self.assertEqual(vars(mode), before)
                self.assertEqual(len(frames), 1)
                self.assertEqual(frames[0].arbitration_id, 0x151)
                self.assertEqual(list(frames[0].data), [1, 255, 10, 0, 0, 2, 1, 0])


if __name__ == '__main__':
    unittest.main()
