import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from core.paths import user_data_root
from robot.settings import SettingsStore


class UserPathTests(unittest.TestCase):
    def test_frozen_windows_uses_user_locallow_not_temp(self):
        local = Path('/Users/example/AppData/Local')
        with patch.dict(os.environ, {'LOCALAPPDATA': str(local), 'TEMP': 'Z:\\unavailable'}), \
                patch.object(sys, 'frozen', True, create=True), patch.object(sys, 'platform', 'win32'):
            expected = local.parent / 'LocalLow' / 'TouchSign'
            self.assertEqual(user_data_root(), expected)
            self.assertEqual(SettingsStore().path, expected / 'config.yaml')

    def test_source_keeps_existing_settings_location(self):
        local = Path('/Users/example/AppData/Local')
        with patch.dict(os.environ, {'LOCALAPPDATA': str(local)}), \
                patch.object(sys, 'frozen', False, create=True), patch.object(sys, 'platform', 'win32'):
            self.assertEqual(user_data_root(), local / 'TouchSign')

    def test_linux_uses_xdg_data_home(self):
        with patch.object(sys, 'platform', 'linux'), patch.dict(os.environ, {'XDG_DATA_HOME': '/tmp/touchsign-data'}):
            self.assertEqual(user_data_root(), Path('/tmp/touchsign-data/TouchSign'))
        with patch.object(sys, 'platform', 'linux'), patch.dict(os.environ, {'XDG_DATA_HOME': 'relative'}):
            self.assertEqual(user_data_root(), Path.home() / '.local/share/TouchSign')

    def test_frozen_runtime_replaces_invalid_temp_for_children(self):
        from gui import runtime
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'LocalLow' / 'TouchSign'
            with patch.object(sys, 'frozen', True, create=True), \
                    patch('gui.runtime.user_data_root', return_value=root), \
                    patch.object(runtime, 'CACHE_DIR', root / '.cache'), \
                    patch.dict(os.environ, {'TEMP': 'Z:\\unavailable', 'TMP': 'Z:\\unavailable'}), \
                    patch.object(tempfile, 'tempdir', None):
                runtime._prepare_cache_dirs()
                self.assertEqual(Path(tempfile.gettempdir()), root / 'temp')
                self.assertEqual(os.environ['TEMP'], str(root / 'temp'))
                self.assertTrue((root / '.cache' / 'matplotlib').is_dir())


if __name__ == '__main__':
    unittest.main()
