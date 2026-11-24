"""Explicit Windows browser launch with observable UI connection feedback."""
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import webbrowser
from contextlib import contextmanager
from core.paths import user_data_root

LOGGER = logging.getLogger(__name__)
_DLL_LOCK = threading.Lock()


def browser_environment():
    env = os.environ.copy()
    bundle = getattr(sys, '_MEIPASS', None)
    if bundle:
        root = os.path.normcase(os.path.abspath(bundle))
        def bundled(part):
            path = os.path.normcase(os.path.abspath(part))
            return path == root or path.startswith(root + os.sep)
        env['PATH'] = os.pathsep.join(p for p in env.get('PATH', '').split(os.pathsep)
                                    if p and not bundled(p))
    return env


@contextmanager
def external_dll_search():
    """PyInstaller's DLL search path must not leak into browser processes."""
    if sys.platform != 'win32' or not getattr(sys, 'frozen', False):
        yield
        return
    import ctypes
    from ctypes import wintypes
    with _DLL_LOCK:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        get_dir = kernel.GetDllDirectoryW
        get_dir.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
        get_dir.restype = wintypes.DWORD
        set_dir = kernel.SetDllDirectoryW
        set_dir.argtypes = [wintypes.LPCWSTR]
        set_dir.restype = wintypes.BOOL
        buffer = ctypes.create_unicode_buffer(get_dir(0, None) + 1)
        get_dir(len(buffer), buffer)
        previous = buffer.value
        if not set_dir(None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not set_dir(previous or None):
                LOGGER.error('Could not restore application DLL search directory')


def browser_candidates():
    roots = [Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')),
             Path(os.environ.get('ProgramFiles', 'C:/Program Files'))]
    for relative in ('Microsoft/Edge/Application/msedge.exe', 'Google/Chrome/Application/chrome.exe'):
        for root in roots:
            path = root / relative
            if path.is_file():
                yield path


def open_browser(url, *, isolated=False):
    if sys.platform == 'win32':
        for executable in browser_candidates():
            try:
                # No shell or file association is required. The UI is explicitly
                # requested by the user, so opening a visible window is intended.
                args = [str(executable), '--new-window', '--no-first-run', '--no-default-browser-check']
                root = user_data_root()
                if isolated:
                    profile = root / 'browser-profile'
                    profile.mkdir(parents=True, exist_ok=True)
                    args.append('--user-data-dir=' + str(profile))
                args.append(url)
                log_dir = root / 'logs'
                log_dir.mkdir(parents=True, exist_ok=True)
                with (log_dir / 'browser.log').open('ab') as output, external_dll_search():
                    process = subprocess.Popen(args, env=browser_environment(),
                                               cwd=str(executable.parent),
                                               stdout=output, stderr=output)
                LOGGER.info('Browser requested: %s pid=%s isolated=%s url=%s', executable, process.pid, isolated, url)
                try:
                    code = process.wait(timeout=2)
                    LOGGER.info('Browser starter exit code=%s', code)
                    if code != 0:
                        continue
                except subprocess.TimeoutExpired:
                    pass
                return True  # UI connection is verified separately.
            except OSError:
                LOGGER.exception('Could not start browser: %s', executable)
    try:
        with external_dll_search():
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
        if connected.wait(10):
            return
        LOGGER.warning('No UI connection; retrying with dedicated browser profile')
        open_browser(url, isolated=True)
        if not connected.wait(25):
            show_address(url)

    def on_startup():
        LOGGER.info('Opening TouchSign UI: %s', url)
        threading.Thread(target=launch, name='touchsign-browser', daemon=True).start()

    app.on_connect(on_connect)
    app.on_startup(on_startup)
