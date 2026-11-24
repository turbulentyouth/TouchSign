"""Explicit Windows browser launch with observable UI connection feedback."""
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import webbrowser

LOGGER = logging.getLogger(__name__)


def browser_candidates():
    roots = [Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')),
             Path(os.environ.get('ProgramFiles', 'C:/Program Files'))]
    for relative in ('Microsoft/Edge/Application/msedge.exe', 'Google/Chrome/Application/chrome.exe'):
        for root in roots:
            path = root / relative
            if path.is_file():
                yield path


def open_browser(url):
    if sys.platform == 'win32':
        for executable in browser_candidates():
            try:
                # No shell or file association is required. The UI is explicitly
                # requested by the user, so opening a visible window is intended.
                process = subprocess.Popen([str(executable), '--new-window', url],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                LOGGER.info('Browser launch: %s pid=%s url=%s', executable, process.pid, url)
                return True
            except OSError:
                LOGGER.exception('Could not start browser: %s', executable)
    try:
        opened = webbrowser.open(url, new=1)
        LOGGER.info('Default browser request result=%s url=%s', opened, url)
        return opened
    except Exception:
        LOGGER.exception('Default browser request failed')
        return False


def show_address(url):
    LOGGER.error('No UI connection after launch; open %s manually', url)
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.MessageBoxW(None,
            f'TouchSign 已启动，但尚未收到界面连接。\n\n请在浏览器地址栏打开：\n{url}\n\n此提示框中按 Ctrl+C 可以复制提示内容。',
            'TouchSign — 打开界面', 0x40 | 0x10000)


def register_browser_launch(app, url):
    connected = threading.Event()

    def on_connect():
        LOGGER.info('TouchSign UI connected: %s', url)
        connected.set()

    def launch():
        # The startup callback runs before uvicorn begins serving requests.
        # Wait briefly in this worker, not in the event loop.
        if connected.wait(.7):
            return
        open_browser(url)
        if not connected.wait(15):
            show_address(url)

    def on_startup():
        LOGGER.info('Opening TouchSign UI: %s', url)
        threading.Thread(target=launch, name='touchsign-browser', daemon=True).start()

    app.on_connect(on_connect)
    app.on_startup(on_startup)
