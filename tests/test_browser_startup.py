import unittest
from unittest.mock import patch, Mock
from pathlib import Path
import tempfile
import os
import sys
from core.config import PROJECT_ROOT, load_config
from gui.runtime import prepare_runtime
prepare_runtime()
from gui.app import run_app


class BrowserStartupTests(unittest.TestCase):
    def test_browser_path_excludes_bundle_and_keeps_system_paths(self):
        from gui.browser_launcher import browser_environment
        with patch.object(sys, '_MEIPASS', 'C:/bundle/_MEI123', create=True), \
                patch.dict(os.environ, {'PATH': 'C:/Windows;C:/bundle/_MEI123;C:/bundle/_MEI123/cv2;C:/other'}):
            self.assertEqual(browser_environment()['PATH'], 'C:/Windows;C:/other')

    def test_dll_search_restored_even_when_child_launch_fails(self):
        from gui.browser_launcher import external_dll_search
        kernel = Mock()
        def get_dir(size, buffer):
            if buffer is not None:
                buffer.value = 'C:/bundle'
            return 9
        kernel.GetDllDirectoryW.side_effect = get_dir
        kernel.SetDllDirectoryW.return_value = True
        with patch.object(sys, 'frozen', True, create=True), patch('ctypes.WinDLL', return_value=kernel):
            with self.assertRaises(OSError):
                with external_dll_search():
                    raise OSError('child failed')
        self.assertEqual([c.args[0] for c in kernel.SetDllDirectoryW.call_args_list], [None, 'C:/bundle'])

    def test_failed_browser_process_tries_next_candidate_with_own_profile(self):
        from gui.browser_launcher import open_browser
        with tempfile.TemporaryDirectory() as tmp, \
                patch('gui.browser_launcher.browser_candidates', return_value=iter([Path('C:/Edge.exe'), Path('C:/Chrome.exe')])), \
                patch('gui.browser_launcher.user_data_root', return_value=Path(tmp)), \
                patch('gui.browser_launcher.external_dll_search'), \
                patch('gui.browser_launcher.subprocess.Popen') as spawn:
            spawn.side_effect = [Mock(pid=1, wait=Mock(return_value=1)), Mock(pid=2, wait=Mock(return_value=0))]
            self.assertTrue(open_browser('http://127.0.0.1:8003/', isolated=True))
            self.assertEqual(spawn.call_count, 2)
            args = spawn.call_args.args[0]
            self.assertIn('--user-data-dir=' + str(Path(tmp)/'browser-profile'), args)
            self.assertNotIn('--no-sandbox', args)
            self.assertTrue((Path(tmp)/'logs/browser.log').is_file())

    def test_release_opens_browser_without_native_window(self):
        config = load_config(PROJECT_ROOT / 'config/raw/config.yaml')
        with patch('gui.app.register_pages'), patch('gui.app.ui.run') as run, \
                patch('gui.browser_launcher.register_browser_launch') as launch, \
                patch('nicegui.native.find_open_port', return_value=8123):
            run_app(config)
        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs['native'])
        self.assertIsNone(kwargs['window_size'])
        self.assertFalse(kwargs['show'])
        self.assertEqual(launch.call_args.args[1], 'http://127.0.0.1:8123/')
        self.assertEqual(kwargs['host'], '127.0.0.1')
        self.assertEqual(kwargs['port'], 8123)

    def test_explicit_port_and_no_show_are_preserved(self):
        config = load_config(PROJECT_ROOT / 'config/raw/config.yaml')
        with patch('gui.app.register_pages'), patch('gui.app.ui.run') as run, \
                patch('nicegui.native.find_open_port') as choose:
            run_app(config, port=9001, show=False)
        choose.assert_not_called()
        self.assertEqual(run.call_args.kwargs['port'], 9001)
        self.assertFalse(run.call_args.kwargs['show'])


if __name__ == '__main__':
    unittest.main()
