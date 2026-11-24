"""Per-user writable paths, separate from the one-file extraction directory."""
import os
from pathlib import Path
import sys


def user_data_root():
    if sys.platform == 'linux':
        configured = os.environ.get('XDG_DATA_HOME', '')
        base = Path(configured) if configured and Path(configured).is_absolute() else Path.home() / '.local' / 'share'
        return base / 'TouchSign'
    local = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local')))
    if getattr(sys, 'frozen', False) and sys.platform == 'win32':
        # Match the Windows bootloader runtime_tmpdir in TouchSign.spec.
        # LocalLow permits ordinary writes even for a low-integrity launch.
        return local.parent / 'LocalLow' / 'TouchSign'
    return local / 'TouchSign'
