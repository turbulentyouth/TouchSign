from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from core.config import PROJECT_ROOT, load_config
from gui.desktop_backend import DesktopBackend
from robot.settings import SettingsStore


class FakeCamera:
    def __init__(self):
        self.serial = 10
        self.sequence = 0
        self.frame = np.full((64, 64, 3), 100, np.uint8)
        self.open = True
    def frame_packet(self):
        if not self.open:
            return None
        self.sequence += 1
        return SimpleNamespace(serial=self.serial, sequence=self.sequence,
                               timestamp=time.monotonic(), frame=self.frame.copy())
    def close_if_owner(self, owner):
        self.open = False


class DesktopBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.events = []
        self.camera = FakeCamera()
        self.patch = patch('gui.desktop_backend.camera_manager', self.camera)
        self.patch.start()
        self.backend = DesktopBackend(load_config(PROJECT_ROOT/'config/raw/config.yaml'), self.events.append,
                                      SettingsStore(Path(self.temp.name)/'config.yaml'))
        self.assertTrue(self.backend.ready.wait(15), 'backend initialization timed out')
        self.command('speech', enabled=False)

    def tearDown(self):
        self.backend.close()
        self.patch.stop()
        self.temp.cleanup()

    def command(self, name, **args):
        return self.backend.handle({'type':'command', 'name':name, **args}).result(timeout=15)

    def wait_until(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.03)
        self.fail('condition timed out')

    def teach_points(self):
        self.command('connect')
        self.command('teach_start')
        self.command('mock_drag', joints=[.1]*7)
        self.command('save_point', point='detect')
        self.command('mock_drag', joints=[0.]*7)
        self.command('save_point', point='home')
        self.command('teach_end')

    def test_calibration_survives_first_frame_and_stream_change_invalidates(self):
        self.wait_until(lambda: self.backend.stream_id == self.camera.serial)
        self.command('calibrate')
        self.assertTrue(self.backend.engine.detector.calibrated)
        time.sleep(.3)
        self.assertTrue(self.backend.engine.detector.calibrated)
        self.camera.serial += 1
        self.wait_until(lambda: not self.backend.engine.detector.calibrated)
        self.command('calibrate')
        self.assertTrue(self.backend.engine.detector.calibrated)
        self.camera.open = False
        self.wait_until(lambda: not self.backend.engine.detector.calibrated)
        self.assertFalse(any(e['type']=='error' for e in self.events), self.events[-5:])

    def test_actual_mock_commands_result_and_close_during_next_cycle(self):
        self.teach_points()
        self.command('detect', key='rat')
        self.assertEqual(self.backend.service.completed, 1)
        self.assertTrue(any(e['type']=='result' and e['key']=='rat' and e['simulated'] for e in self.events))
        self.assertFalse(any(e['type']=='error' for e in self.events), self.events[-5:])
        task = self.backend.handle({'type':'command','name':'detect','key':'ox'})
        self.wait_until(lambda: self.backend.service.busy)
        self.backend.close()
        self.assertTrue(task.done())
        self.assertFalse(self.backend._thread.is_alive())
        self.assertIsNone(self.backend.service.arm)
        self.assertTrue(all(not t.is_alive() for t in self.backend.service.pool._threads))
        self.assertTrue(all(not t.is_alive() for t in self.backend._vision._threads))
        self.assertTrue(any(e['type']=='closed' for e in self.events))


if __name__ == '__main__':
    unittest.main()
