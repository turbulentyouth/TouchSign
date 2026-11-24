import unittest
from unittest.mock import patch
from core.config import PROJECT_ROOT, load_config
from gui.runtime import prepare_runtime
prepare_runtime()
from gui.app import run_app


class BrowserStartupTests(unittest.TestCase):
    def test_release_opens_browser_without_native_window(self):
        config = load_config(PROJECT_ROOT / 'config/raw/config.yaml')
        with patch('gui.app.register_pages'), patch('gui.app.ui.run') as run, \
                patch('nicegui.native.find_open_port', return_value=8123):
            run_app(config)
        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs['native'])
        self.assertIsNone(kwargs['window_size'])
        self.assertTrue(kwargs['show'])
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
