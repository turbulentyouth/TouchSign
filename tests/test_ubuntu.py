from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from robot.nero import NeroArm
from robot.settings import RobotSettings
from robot.transport import can_interface, check_socketcan


class UbuntuTests(unittest.TestCase):
    def test_platform_selects_transport_and_default_channel(self):
        for system, backend, channel in [('linux', 'socketcan', 'can0'), ('win32', 'agx_cando', '0')]:
            with self.subTest(system=system), patch.object(sys, 'platform', system):
                self.assertEqual(can_interface(), backend)
                self.assertEqual(RobotSettings().channel, channel)

    def test_can_preflight_rejects_missing_wrong_or_down_interfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                check_socketcan('can0', root)
            with self.assertRaises(ValueError):
                check_socketcan('../can0', root)
            device = root / 'can0'
            device.mkdir()
            (device / 'type').write_text('1')
            (device / 'flags').write_text('0x1')
            with self.assertRaises(ValueError):
                check_socketcan('can0', root)
            (device / 'type').write_text('280')
            (device / 'flags').write_text('0x80')
            with self.assertRaisesRegex(RuntimeError, '未启用'):
                check_socketcan('can0', root)
            (device / 'flags').write_text('0x81')
            check_socketcan('can0', root)

    def test_connect_uses_socketcan_without_enabling_or_moving(self):
        probe, robot = Mock(), Mock()
        probe.get_firmware.return_value = {'software_version': '1.21'}
        with patch.object(sys, 'platform', 'linux'), patch('robot.nero.check_socketcan') as check, \
                patch('pyAgxArm.create_agx_arm_config', return_value={}) as config, \
                patch('pyAgxArm.AgxArmFactory.create_arm', side_effect=[probe, robot]):
            arm = NeroArm(RobotSettings(channel='can_follower'))
            arm.connect()
            check.assert_called_once_with('can_follower')
            self.assertEqual(config.call_count, 2)
            for call in config.call_args_list:
                self.assertEqual(call.kwargs['interface'], 'socketcan')
                self.assertEqual(call.kwargs['channel'], 'can_follower')
                self.assertFalse(call.kwargs['auto_connect'])
            for device in (probe, robot):
                device.enable.assert_not_called()
                device.move_j.assert_not_called()
                device.set_leader_mode.assert_not_called()
            probe.disconnect.assert_called_once()
            arm.disconnect()
            robot.disconnect.assert_called_once()

    def test_linux_desktop_routes_to_browser(self):
        from desktop_main import _run_ui
        from core.config import Config
        args = SimpleNamespace(self_test_report=None, no_show=True, port=8088)
        with tempfile.TemporaryDirectory() as tmp, patch('core.paths.user_data_root', return_value=Path(tmp)), \
                patch('core.instance.desktop_instance', return_value=nullcontext(None)), \
                patch.object(sys, 'platform', 'linux'), patch('gui.runtime.prepare_runtime'), \
                patch('gui.app.run_app') as run:
            config = Config()
            _run_ui(config, args)
            run.assert_called_once_with(config, native=False, show=False, port=8088)

    def test_linux_discovers_sparse_camera_indices(self):
        from core.camera import scan_cameras, _backends
        import cv2
        with patch.object(sys, 'platform', 'linux'), \
                patch('core.linux_video.is_color_capture', return_value=True), \
                patch('core.camera.Path.glob', return_value=[Path('/dev/video12'), Path('/dev/video2')]), \
                patch('core.camera._probe_camera', side_effect=lambda i: i) as probe:
            self.assertEqual(_backends(), ((cv2.CAP_V4L2, 'V4L2'),))
            self.assertEqual(scan_cameras(max_index=6), [2, 12])
            self.assertEqual([call.args[0] for call in probe.call_args_list], [2, 12])

    def test_configured_sensor_does_not_fall_back_to_webcam(self):
        from core.camera import CameraInfo, preferred_camera
        webcam = CameraInfo(0, 'JYU2C-2083')
        sensor = CameraInfo(8, 'VBTS: IXM-STD')
        self.assertIs(preferred_camera([webcam, sensor], 'VBTS'), sensor)
        self.assertIsNone(preferred_camera([webcam], 'VBTS'))
        self.assertIs(preferred_camera([webcam]), webcam)
        # After replugging, the old sensor index may belong to a webcam.
        self.assertIs(preferred_camera([webcam, sensor], 'VBTS', previous_index=0), sensor)
        self.assertIs(preferred_camera([webcam, sensor], previous_index=8), sensor)


if __name__ == '__main__':
    unittest.main()
