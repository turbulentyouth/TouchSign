"""运行时兼容层：在启动 NiceGUI 之前准备好解释器环境。

当前开发机开启了 Windows 智能应用控制（Smart App Control），uv 下载的独立
CPython 所附带的 ``libssl-3-x64.dll`` 未签名，会被系统策略拦截，导致
``import ssl`` 失败。本模块会在检测到该问题时，加载系统中已签名的 OpenSSL 3
DLL（例如 Microsoft Copilot 自带的副本）作为兼容桥接。

它同时把 Matplotlib 缓存重定向到项目内的可写目录，并关闭 NiceGUI 中当前界面
用不到的进程池，以便在受限环境（包括 DSH 沙箱）里也能直接启动。
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ((Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'TouchSign' / '.cache')
             if getattr(sys, 'frozen', False) else PROJECT_ROOT / '.cache')

_DLL_DIR_HANDLES: list[object] = []
_SSL_READY = False


def prepare_runtime() -> None:
    """在任何 NiceGUI / GUI 导入之前调用。"""
    global _SSL_READY

    _prepare_cache_dirs()
    _ensure_ssl()
    _configure_nicegui()
    _SSL_READY = True


def ssl_ready() -> bool:
    """是否已成功准备 SSL（供启动日志使用）。"""
    return _SSL_READY


def _prepare_cache_dirs() -> None:
    matplotlib_dir = CACHE_DIR / 'matplotlib'
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(matplotlib_dir))


def _openssl_candidates() -> list[Path]:
    candidates: list[Path] = []

    configured = os.environ.get('TOUCHSIGN_OPENSSL_DIR')
    if configured:
        candidates.append(Path(configured))

    candidates.append(PROJECT_ROOT / 'runtime' / 'openssl')

    for env_name in ('ProgramFiles(x86)', 'ProgramFiles', 'ProgramW6432'):
        program_files = os.environ.get(env_name)
        if program_files:
            base = Path(program_files)
            candidates.append(base / 'Microsoft' / 'Copilot' / 'Application' / 'undocked_copilot')
            candidates.append(base / 'Git' / 'mingw64' / 'bin')
            candidates.append(base / 'Git' / 'mingw64' / 'libexec' / 'git-core')

    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item).lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _try_load_openssl(directory: Path) -> bool:
    """尝试从目录加载一对 OpenSSL 3 DLL；成功返回 True。"""
    libcrypto = directory / 'libcrypto-3-x64.dll'
    libssl = directory / 'libssl-3-x64.dll'
    if not (libcrypto.is_file() and libssl.is_file()):
        return False

    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
        ctypes.WinDLL(str(libcrypto))
        ctypes.WinDLL(str(libssl))
    except OSError:
        return False
    return True


def _ensure_ssl() -> None:
    try:
        import ssl  # noqa: F401
        return
    except ImportError:
        pass

    if sys.platform != 'win32':
        raise RuntimeError('Python 的 ssl 模块不可用，无法启动 NiceGUI（非 Windows 平台）。')

    for directory in _openssl_candidates():
        if not _try_load_openssl(directory):
            continue
        try:
            import ssl  # noqa: F401
            return
        except ImportError:
            continue

    raise RuntimeError(
        '无法加载 Python 的 ssl 模块。当前系统的“智能应用控制”（Smart App Control）'
        '可能拦截了 uv 自带 Python 的未签名 OpenSSL DLL。\n'
        '建议使用 python.org 官方安装的 Python 3.12 重建环境，或\n'
        '把已签名的 libssl-3-x64.dll / libcrypto-3-x64.dll 放入 '
        'runtime/openssl/ 或设置 TOUCHSIGN_OPENSSL_DIR。'
    )


def _configure_nicegui() -> None:
    """关闭 NiceGUI 的进程池。

    NiceGUI 默认创建 ProcessPoolExecutor，供 ``run.cpu_bound`` 使用。当前页面
    只使用定时器与轻量计算，不需要进程池；关闭后可避免在受限环境中因命名管道
    被拒绝而无法启动。
    """
    if os.environ.get('TOUCHSIGN_PROCESS_POOL') == '1':
        return
    try:
        import nicegui.run
    except ImportError:
        return
    nicegui.run.setup = lambda: None
