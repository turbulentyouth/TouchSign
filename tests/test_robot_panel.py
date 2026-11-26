from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from gui.runtime import prepare_runtime
prepare_runtime()
from core.config import Config
from core.camera import FramePacket
from gui.app import TouchSignView
from robot.settings import SettingsStore, RobotSettings
from robot.service import RobotService


class RobotPanelTests(unittest.TestCase):
    def test_detection_passes_configured_fifty_percent_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            service = RobotService(RobotSettings(), SettingsStore(Path(directory)/'config.yaml'))
            self.addCleanup(service.pool.shutdown, wait=True)
            config = Config()
            config = replace(config, app=replace(config.app, robot_controls=True),
                             recognition=replace(config.recognition, confidence_threshold=.5))
            with patch('gui.app.ui.timer'), patch('gui.app.camera_manager') as cameras, \
                    patch('gui.robot_panel._service', service), \
                    patch('gui.robot_panel.camera_manager.frame_packet', return_value=None), \
                    patch.object(service, 'start') as start:
                cameras.cameras = []
                view = TouchSignView(config)
                coroutine = view.robot_panel.detect()
                try:
                    with self.assertRaises(StopIteration):
                        coroutine.send(None)
                finally:
                    coroutine.close()
                self.assertEqual(start.call_args.kwargs['confidence_threshold'], .5)

    def test_tactile_processing_starts_only_after_jaw_arrival(self):
        async def immediate_io(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch('gui.app.ui.timer'), patch('gui.app.camera_manager') as cameras:
            cameras.cameras = []
            config = Config()
            view = TouchSignView(replace(config, app=replace(config.app, robot_controls=False)))
            service = SimpleNamespace(mock=False, phase='gripping', recognition_start=time.monotonic())
            panel = SimpleNamespace(service=service)
            service.owner = panel
            view.robot_panel = panel
            view.streaming = view.connected = True
            view._pending_reset = False
            view._stream_serial = 7
            process = Mock(return_value=None)
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            phases = [('gripping', time.monotonic(), 0),
                      ('recognizing', service.recognition_start-.1, 0),
                      ('recognizing', time.monotonic(), 1),
                      ('releasing', time.monotonic(), 2)]
            with patch('gui.app.run.io_bound', immediate_io), \
                    patch.object(view.engine, 'process', process), \
                    patch('gui.app.LOGGER.exception') as error:
                for sequence, (phase, timestamp, expected_calls) in enumerate(phases, 1):
                    service.phase = phase
                    cameras.frame_packet.return_value = FramePacket(7, sequence, timestamp, frame)
                    coroutine = view._tick()
                    try:
                        with self.assertRaises(StopIteration):
                            coroutine.send(None)
                    finally:
                        coroutine.close()
                    self.assertEqual(process.call_count, expected_calls)
                error.assert_not_called()

    def test_offline_controls_build_and_do_not_connect(self):
        with tempfile.TemporaryDirectory() as directory:
            service=RobotService(RobotSettings(),SettingsStore(Path(directory)/'config.yaml'))
            config=Config()
            config=replace(config,app=replace(config.app,robot_controls=True))
            with patch('gui.app.ui.timer'), patch('gui.app.camera_manager') as cameras, \
                 patch('gui.robot_panel._service',service):
                cameras.cameras=[]
                view=TouchSignView(config)
                panel=view.robot_panel
                self.assertEqual(panel.mode.value,'mock')
                self.assertIsNone(service.arm)
                self.assertEqual(panel.detect_button.text,'检测（运行一轮）')
                self.assertFalse(panel.detect_button.enabled)
                self.assertIs(panel.detect_button, view.detect_button)
                self.assertIs(panel.stop_button, view.robot_stop_button)
                self.assertEqual(panel.stop_button.text, '停止本轮并收回')
                self.assertEqual(panel.close_width.label, '闭合目标间距 mm')
                self.assertFalse(hasattr(panel, 'grip_step'))
                self.assertEqual(panel.close_speed.label, '夹爪闭合限速 mm/s')
                self.assertEqual(panel.close_speed.value, 5)
                self.assertEqual(panel.close_speed._props['max'], 30)
                self.assertFalse(view.robot_settings.visible)
                view._toggle_robot_settings()
                self.assertTrue(view.robot_settings.visible)
                view._toggle_robot_settings()
                self.assertFalse(view.robot_settings.visible)
                self.assertTrue(view.exit_button.visible)
                self.assertFalse(service.store.path.exists())
                self.assertFalse(panel.real_ack.value)
                self.assertIn('离线模拟',panel.notice.text)
                service.owner = panel
                service.phase = 'gripping'
                service.message = '等待夹爪到位：目标 3.5 mm'
                panel.refresh()
                self.assertEqual(view.runtime_hint.text, service.message)
                service.busy = True
                panel.refresh()
                self.assertFalse(panel.close_speed.enabled)
                service.busy = False
                service.arm = type('Arm', (), {'profile':'mock'})()
                service.settings.points['mock'] = {'detect':[.1]*7, 'home':[0.]*7}
                service.phase = 'releasing'
                service.finishing = panel.action_busy = True
                panel.refresh()
                self.assertFalse(panel.stop_button.enabled)
                self.assertTrue(panel.disconnect_button.enabled)
                service.finishing = panel.action_busy = False
                service.phase = 'complete'
                service.message = '本轮已停止，夹爪已张开并返回收回点'
                panel.refresh()
                self.assertTrue(panel.detect_button.enabled)
                self.assertEqual(view.runtime_hint.text, service.message)
                for speed in (2, 30):
                    panel.close_speed.value = speed
                    panel.apply_settings()
                    self.assertEqual(service.store.load().gripper_close_speed, speed/1000)
                service.arm = None
                service.phase = 'disabled'
                panel.refresh()
                self.assertTrue(panel.teach_button.enabled)
                self.assertFalse(panel.detect_button.enabled)
                service.phase = 'fault'
                panel.refresh()
                self.assertFalse(panel.teach_button.enabled)
                service.phase = 'stopped'
                service.message = '已停止本轮；后续收回已取消'
                panel.refresh()
                self.assertFalse(panel.teach_button.enabled)
                self.assertFalse(panel.detect_button.enabled)
                self.assertEqual(view.runtime_hint.text, service.message)
            service.pool.shutdown(wait=True)
