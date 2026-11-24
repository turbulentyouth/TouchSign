"""Embedded WebView2 with native callbacks: no server or custom child EXE."""
import json
import logging
from pathlib import Path
import tempfile
from core.config import PROJECT_ROOT
from core.paths import user_data_root
from robot.settings import SettingsStore
from gui.desktop_backend import DesktopBackend
from gui.win32_webview import Win32WebView

LOGGER = logging.getLogger(__name__)


def run_desktop(config, self_test_report=None):
    root = user_data_root()
    temporary = tempfile.TemporaryDirectory(prefix='touchsign-gui-test-', ignore_cleanup_errors=True) if self_test_report else None
    profile = Path(temporary.name) / 'webview2' if temporary else root / 'webview2-desktop'
    backend = None
    def received(message):
        if message.get('type') == 'command':
            backend.handle(message)
        else:
            LOGGER.info('Native window event: %s', message)
    window = Win32WebView(html_path=PROJECT_ROOT / 'gui/desktop.html', profile=profile,
                         on_message=received, report=self_test_report,
                         selftest_script=PROJECT_ROOT / 'gui/desktop_acceptance.js' if self_test_report else None)
    def send(message):
        if message.get('type') == 'close_requested':
            window.report['exit_button_clicked'] = True
            window.close()
        elif not window.finished:
            window.post(message)
    try:
        store = SettingsStore(Path(temporary.name) / 'config.yaml') if temporary else None
        backend = DesktopBackend(config, send, settings_store=store)
        result = window.run()
    finally:
        if backend:
            backend.close()
        if temporary:
            temporary.cleanup()
    if self_test_report:
        result.update(backend_closed=bool(backend and backend.closed),
                      backend_thread_stopped=bool(backend and not backend._thread.is_alive()),
                      transport='native-message-callbacks', http_server=False, external_browser=False,
                      pythonnet=False, custom_native_executable=False)
        result['passed'] = bool(result.get('passed') and result['backend_thread_stopped'] and result.get('exit_button_clicked'))
        Path(self_test_report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
