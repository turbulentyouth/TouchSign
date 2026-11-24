import errno
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.instance import desktop_instance
from core.linux_video import is_color_capture, VIDIOC_QUERYCAP


@unittest.skipUnless(sys.platform == 'linux', 'Linux device and lock APIs')
class LinuxDevicesTests(unittest.TestCase):
    def test_query_uses_node_caps_and_skips_metadata(self):
        def ioctl(fd, request, data):
            self.assertEqual(request, VIDIOC_QUERYCAP)
            # The physical camera captures video, but this node is metadata only.
            struct.pack_into('=II', data, 84, 0x80000001, 0x00800000)
        with patch('os.open', return_value=17), patch('os.close') as close, patch('fcntl.ioctl', side_effect=ioctl):
            self.assertFalse(is_color_capture(9))
            close.assert_called_once_with(17)

    def test_formats_separate_depth_from_color(self):
        for formats, expected in [([b'Z16 '], False), ([b'Y16 '], False), ([b'YUYV', b'MJPG'], True)]:
            def ioctl(fd, request, data):
                if request == VIDIOC_QUERYCAP:
                    struct.pack_into('=II', data, 84, 0x80000001, 1)
                else:
                    index = struct.unpack_from('=I', data)[0]
                    if index >= len(formats):
                        raise OSError(errno.EINVAL, 'end of formats')
                    data[44:48] = formats[index]
            with self.subTest(formats=formats), patch('os.open', return_value=17), \
                    patch('os.close'), patch('fcntl.ioctl', side_effect=ioctl):
                self.assertEqual(is_color_capture(8), expected)

    def test_busy_sensor_is_reported_instead_of_disappearing(self):
        from core.camera import scan_cameras
        with patch('core.camera.Path.glob', return_value=[Path('/dev/video8')]), \
                patch('core.linux_video.is_color_capture', return_value=True), \
                patch('core.camera._camera_name', return_value='VBTS'), \
                patch('core.camera._probe_camera', return_value=None):
            cameras = scan_cameras()
            self.assertEqual(len(cameras), 1)
            self.assertFalse(cameras[0].available)
            self.assertIn('占用', cameras[0].error)

    def test_lock_blocks_second_instance_and_recovers_after_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'desktop.lock'
            with self.assertRaisesRegex(RuntimeError, 'exit'):
                with desktop_instance(path, 'http://127.0.0.1:8080/') as first:
                    self.assertIsNone(first)
                    with desktop_instance(path, 'http://127.0.0.1:8000/') as second:
                        self.assertEqual(second['url'], 'http://127.0.0.1:8080/')
                    raise RuntimeError('exit')
            with desktop_instance(path, 'http://127.0.0.1:9000/') as restarted:
                self.assertIsNone(restarted)

    def test_named_sensor_scan_never_opens_other_cameras(self):
        from core.camera import scan_cameras, CameraInfo
        with patch('core.camera.Path.glob', return_value=[Path('/dev/video0'), Path('/dev/video8')]), \
                patch('core.camera._camera_name', side_effect=lambda i: 'Webcam' if i == 0 else 'VBTS'), \
                patch('core.linux_video.is_color_capture', return_value=True) as capability, \
                patch('core.camera._probe_camera', return_value=CameraInfo(8, 'VBTS')) as probe:
            self.assertEqual(scan_cameras(name_filter='VBTS')[0].index, 8)
            capability.assert_called_once_with(8)
            probe.assert_called_once_with(8)

    def test_duplicate_desktop_exits_before_scanning_and_opens_existing_url(self):
        from desktop_main import _run_ui
        from core.config import Config
        with tempfile.TemporaryDirectory() as tmp, \
                patch('core.paths.user_data_root', return_value=Path(tmp)), \
                patch('gui.app.run_app') as run, patch('gui.browser_launcher.open_browser') as browser:
            with desktop_instance(Path(tmp) / 'desktop.lock', 'http://127.0.0.1:8080/'):
                _run_ui(Config(), SimpleNamespace(self_test_report=None, no_show=False, port=8000))
                run.assert_not_called()
                browser.assert_called_once_with('http://127.0.0.1:8080/')


if __name__ == '__main__':
    unittest.main()
