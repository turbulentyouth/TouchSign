from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gui.runtime import prepare_runtime
prepare_runtime()
from core.config import Config
from gui.app import TouchSignView
from robot.settings import SettingsStore, RobotSettings
from robot.service import RobotService


class RobotPanelTests(unittest.TestCase):
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
                self.assertFalse(view.robot_settings.visible)
                view._toggle_robot_settings()
                self.assertTrue(view.robot_settings.visible)
                view._toggle_robot_settings()
                self.assertFalse(view.robot_settings.visible)
                self.assertTrue(view.exit_button.visible)
                self.assertFalse(service.store.path.exists())
                self.assertFalse(panel.real_ack.value)
                self.assertIn('离线模拟',panel.notice.text)
            service.pool.shutdown(wait=True)
