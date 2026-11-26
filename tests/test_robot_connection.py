import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from robot.service import RobotService
from robot.settings import RobotSettings, SettingsStore
from tests.test_robot_cycle import OfflineLoop


class RobotConnectionTests(unittest.IsolatedAsyncioTestCase):
    def _setupAsyncioRunner(self):
        self._asyncioRunner = asyncio.Runner(loop_factory=OfflineLoop, debug=True)

    async def test_connection_failure_stays_visible_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            service = RobotService(RobotSettings(backend='nero', channel='can_follower'),
                                   SettingsStore(Path(directory) / 'config.yaml'))
            failed = Mock()
            failed.connect.side_effect = RuntimeError('无法读取 NERO 固件')
            owner = object()
            try:
                with patch('robot.nero.NeroArm', return_value=failed), \
                        self.assertLogs('robot.service', level='ERROR') as logs, \
                        self.assertRaisesRegex(RuntimeError, '无法读取 NERO 固件'):
                    await service.connect(owner)
                self.assertEqual(service.phase, 'disconnected')
                self.assertEqual(service.message, '连接失败：无法读取 NERO 固件')
                self.assertIn('can_follower', logs.output[0])
                self.assertIsNone(service.owner)
                self.assertIsNone(service.arm)
                self.assertFalse(service.busy)
                self.assertIsNone(service.monitor_task)
                failed.disconnect.assert_called_once()
                failed.enable.assert_not_called()
                failed.move_j.assert_not_called()
                # Retry must clear the stale failure and resume normal monitoring.
                service.settings.backend = 'mock'
                await service.connect(owner)
                self.assertEqual(service.phase, 'idle')
                self.assertEqual(service.message, '模拟设备已连接')
                await service.disconnect(owner)
            finally:
                service.pool.shutdown(wait=True)

    async def test_cleanup_failure_preserves_connection_error_and_releases_owner(self):
        service = RobotService(RobotSettings(backend='nero'), Mock())
        failed = Mock()
        failed.connect.side_effect = TimeoutError('固件查询超时')
        failed.disconnect.side_effect = RuntimeError('关闭通信失败')
        try:
            with patch('robot.nero.NeroArm', return_value=failed), \
                    self.assertLogs('robot.service', level='ERROR'), \
                    self.assertRaisesRegex(TimeoutError, '固件查询超时'):
                await service.connect(object())
            self.assertEqual(service.message, '连接失败：固件查询超时')
            self.assertIsNone(service.owner)
            self.assertIsNone(service.arm)
            self.assertFalse(service.busy)
        finally:
            service.pool.shutdown(wait=True)
