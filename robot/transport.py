"""Platform-specific CAN selection; inspecting an interface never enables it."""
from pathlib import Path
import re
import sys


def can_interface():
    if sys.platform == 'linux':
        return 'socketcan'
    if sys.platform == 'win32':
        return 'agx_cando'
    raise RuntimeError('机械臂目前支持 Ubuntu SocketCAN 和 Windows USB-CAN')


def check_socketcan(channel, root=Path('/sys/class/net')):
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,15}', channel):
        raise ValueError('CAN 通道必须是网口名称，例如 can0 或 can_follower')
    device = root / channel
    try:
        kind = (device / 'type').read_text().strip()
        flags = int((device / 'flags').read_text().strip(), 16)
    except OSError as exc:
        raise RuntimeError(f'找不到 CAN 接口 {channel}，请检查 USB-CAN 连接及网口名称') from exc
    if kind != '280':  # ARPHRD_CAN
        raise ValueError(f'{channel} 不是 CAN 接口')
    if not flags & 1:  # IFF_UP; CAN operstate may be "unknown" even when working.
        raise RuntimeError(f'CAN 接口 {channel} 未启用，请按 docs/ubuntu.md 配置波特率并启用')
