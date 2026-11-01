"""触象 · TouchSign 启动入口。

启动前先调用 ``prepare_runtime``，处理 Windows 智能应用控制对未签名 OpenSSL
DLL 的拦截，并把缓存重定向到项目目录；随后加载配置并启动 NiceGUI。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import multiprocessing
import os
from pathlib import Path
import sys
import traceback


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from core.config import DEFAULT_CONFIG_PATH

    parser = argparse.ArgumentParser(
        prog='TouchSign',
        description='触象 · 视触觉生肖识别系统',
    )
    parser.add_argument(
        '--config',
        default=str(DEFAULT_CONFIG_PATH),
        help='YAML 配置文件路径（默认 config/config.yaml）',
    )
    parser.add_argument('--host', help='监听地址，覆盖配置文件')
    parser.add_argument('--port', type=int, help='监听端口，覆盖配置文件')
    parser.add_argument(
        '--preview-model', metavar='ONNX', help='在 GUI 预览指定 ONNX 的首选类别，跳过拒识阈值',
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='仅启动后台服务，不显示窗口或打开浏览器',
    )
    parser.add_argument(
        '--native',
        action='store_true',
        help='使用 pywebview 原生窗口（触摸屏全屏场景）',
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='开发模式：文件变化时自动重载',
    )
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='列出可用摄像头的设备编号后退出',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        help='日志级别（默认 INFO）',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from core.config import load_config

    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    )
    config = load_config(args.config)

    if args.list_cameras:
        from core.camera import describe_cameras, scan_cameras

        cameras = scan_cameras(max_index=config.sensor.scan_max_index)
        print(describe_cameras(cameras))
        return

    from gui.app import run_app

    if args.preview_model:
        config = replace(config, recognition=replace(
            config.recognition, model_path=args.preview_model, manifest_path='', preview=True,
        ))
    run_app(
        config,
        host=args.host,
        port=args.port,
        show=False if args.no_show else None,
        reload=True if args.reload else None,
        native=(args.native or config.app.native) and not args.no_show,
    )


def _launch() -> None:
    frozen = bool(getattr(sys, 'frozen', False))
    log_path = None
    try:
        if frozen:
            candidate_log = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'TouchSign' / 'logs' / 'startup.log'
            candidate_log.parent.mkdir(parents=True, exist_ok=True)
            # Windowed EXEs have no stdout/stderr. Uvicorn and spawned native
            # windows need real streams for logging and startup diagnostics.
            output = candidate_log.open('a', encoding='utf-8', buffering=1)
            log_path = candidate_log
            sys.stdout = output
            sys.stderr = output
        from gui.runtime import prepare_runtime

        prepare_runtime()
        multiprocessing.freeze_support()
        main()
    except KeyboardInterrupt:
        pass
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and not exc.code:
            return
        if not frozen:
            raise
        detail = traceback.format_exc()
        if sys.stderr is not None:
            try:
                sys.stderr.write(detail)
                sys.stderr.flush()
            except OSError:
                pass
        import ctypes

        message = f'TouchSign 启动失败\n\n{exc}'
        if log_path is not None:
            message += f'\n\n诊断日志\n{log_path}'
        ctypes.windll.user32.MessageBoxW(None, message, 'TouchSign', 0x10)
        raise SystemExit(1) from None


if __name__ == '__main__':
    _launch()
elif __name__ == '__mp_main__':
    from gui.runtime import prepare_runtime

    prepare_runtime()
