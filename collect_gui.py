"""Ubuntu collection console; independent from desktop_main.py and collect.py."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
import logging
from pathlib import Path
import sys


def parse_args(argv=None):
    from core.config import PROJECT_ROOT

    parser = argparse.ArgumentParser(description='触象 · 复用现有 GUI 和机械臂流程的数据采集')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config/ubuntu/config.yaml')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/raw', help='采集数据根目录')
    parser.add_argument('--notes', default='', help='空载校准后写入会话的备注，可在 GUI 修改')
    parser.add_argument('--samples-per-cycle', type=int, default=1, help='每轮目标张数，默认 1（1–20）')
    parser.add_argument('--interval', type=float, default=.5, help='自动采集最短间隔，默认 0.5 秒')
    parser.add_argument('--port', type=int, default=8081, help='本地服务端口，默认 8081')
    parser.add_argument('--no-show', action='store_true', help='仅启动服务，不打开浏览器')
    parser.add_argument('--summary', action='store_true', help='输出已有数据统计，不连接硬件')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.samples_per_cycle <= 20 or not .1 <= args.interval <= 60:
        parser.error('端口须为 1–65535，每轮张数须为 1–20，间隔须为 0.1–60 秒')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.summary:
        from core.dataset import dataset_summary
        print(json.dumps(dataset_summary(args.output.expanduser()), ensure_ascii=False, indent=2))
        return 0
    from core.config import load_config
    from core.instance import desktop_instance
    from core.paths import user_data_root

    config = load_config(args.config)
    config = replace(config, app=replace(config.app, title='触象 · TouchSign | 数据采集与标注'))
    root = user_data_root()
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        filename=root / 'logs/collection.log', encoding='utf-8')
    host = '127.0.0.1' if config.app.host == '0.0.0.0' else config.app.host
    # Share the original process lock: two commands must not compete for CAN/camera.
    with desktop_instance(root / 'desktop.lock', f'http://{host}:{args.port}/') as existing:
        if existing is not None:
            print('TouchSign 已在运行，请先退出原程序，再启动数据采集界面。', file=sys.stderr)
            return 1
        from gui.runtime import prepare_runtime
        prepare_runtime()
        from gui.app import run_app
        from gui.collection_app import CollectionView
        factory = partial(CollectionView, output=args.output, notes=args.notes,
                          target=args.samples_per_cycle, interval=args.interval)
        run_app(config, native=False, show=not args.no_show, port=args.port, reload=False,
                view_factory=factory)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logging.exception('采集界面启动失败')
        print(f'采集界面启动失败：{exc}', file=sys.stderr)
        raise SystemExit(1)
