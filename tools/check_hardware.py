"""Ubuntu device inventory and optional bounded receive checks; never sends CAN."""
import argparse
import json
from pathlib import Path
import platform
import time

from robot.transport import check_socketcan


def inventory():
    interfaces = []
    for path in sorted(Path('/sys/class/net').iterdir()):
        try:
            if (path / 'type').read_text().strip() != '280':
                continue
            interfaces.append({'name': path.name,
                               'up': bool(int((path / 'flags').read_text().strip(), 16) & 1)})
        except OSError:
            continue
    videos = []
    for path in sorted(Path('/sys/class/video4linux').glob('video*')):
        try:
            videos.append({'device': '/dev/' + path.name, 'name': (path / 'name').read_text().strip()})
        except OSError:
            continue
    return {'system': platform.platform(), 'can': interfaces, 'video': videos,
            'can_commands_sent': False, 'errors': []}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Ubuntu 硬件诊断：只读清单，可选读取图像或被动接收 CAN')
    parser.add_argument('--camera', type=int, help='读取指定摄像头的一帧，不保存图像')
    parser.add_argument('--can-channel', help='被动接收已启用的 CAN 网口，不发送指令')
    parser.add_argument('--report', type=Path, help='另存 JSON 报告')
    args = parser.parse_args(argv)
    report = inventory()
    if args.camera is not None:
        from core.camera import _probe_camera
        from dataclasses import asdict
        camera = _probe_camera(args.camera)
        report['camera_read'] = asdict(camera) if camera else None
        if camera is None:
            report['errors'].append(f'摄像头 {args.camera} 无法读取')
    if args.can_channel:
        try:
            check_socketcan(args.can_channel)
            import can
            deadline = time.monotonic() + 2
            identifiers = set()
            count = 0
            with can.Bus(interface='socketcan', channel=args.can_channel,
                         receive_own_messages=False, ignore_config=True) as bus:
                while (remaining := deadline - time.monotonic()) > 0:
                    message = bus.recv(timeout=remaining)
                    if message is not None and not message.is_error_frame:
                        count += 1
                        identifiers.add(hex(message.arbitration_id))
            report['can_receive'] = {'channel': args.can_channel, 'frames': count,
                                     'ids': sorted(identifiers)}
            if not count:
                report['errors'].append('两秒内未收到 CAN 数据，尚不能确认机械臂在线')
        except Exception as exc:
            report['errors'].append(str(exc))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + '\n', encoding='utf-8')
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
