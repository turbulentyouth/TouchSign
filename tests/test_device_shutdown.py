import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from gui.runtime import prepare_runtime
prepare_runtime()
from gui.app import _mjpeg_stream, _shutdown_devices
from tests.test_robot_cycle import OfflineLoop


class DeviceShutdownTests(unittest.IsolatedAsyncioTestCase):
    def _setupAsyncioRunner(self):
        self._asyncioRunner = asyncio.Runner(loop_factory=OfflineLoop, debug=True)

    async def test_camera_closes_even_when_robot_cleanup_fails(self):
        with patch('gui.robot_panel.shutdown_robot', new_callable=AsyncMock) as robot, \
                patch('gui.app.camera_manager') as camera:
            robot.side_effect = RuntimeError('transport failure')
            with self.assertLogs('gui.app', level='ERROR'):
                await _shutdown_devices(True)
            camera.close.assert_called_once()

    async def test_stream_finishes_on_server_exit_with_or_without_frames(self):
        for jpeg in (None, b'jpeg'):
            with self.subTest(jpeg=jpeg), \
                    patch('gui.app.Server.instance', SimpleNamespace(should_exit=False), create=True) as server, \
                    patch('gui.app.camera_manager') as camera:
                camera.snapshot.return_value = (1, 1, jpeg)
                stream = _mjpeg_stream()
                if jpeg:
                    self.assertIn(jpeg, await anext(stream))
                pending = asyncio.create_task(anext(stream))
                await asyncio.sleep(.05)
                server.should_exit = True
                with self.assertRaises(StopAsyncIteration):
                    await asyncio.wait_for(pending, .5)
                await stream.aclose()
