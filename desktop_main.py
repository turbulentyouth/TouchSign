"""TouchSign entry point: Windows WebView2 or Ubuntu local browser UI."""
from pathlib import Path
import argparse
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import traceback
from contextlib import nullcontext


def _run_ui(config, args):
    if sys.platform == 'win32':
        from gui.embedded import run_desktop
        run_desktop(config, self_test_report=args.self_test_report)
    else:
        if args.self_test_report:
            raise ValueError('--self-test-report 是 Windows WebView2 验收；Ubuntu 离线自检请用 --diagnostics-report')
        port = args.port if args.port is not None else config.app.port
        if not 1 <= port <= 65535:
            raise ValueError('端口必须在 1–65535 之间')
        browser_host = config.app.host if config.app.host != '0.0.0.0' else '127.0.0.1'
        url = f'http://{browser_host}:{port}/'
        guard = nullcontext(None)
        if sys.platform == 'linux':
            from core.instance import desktop_instance
            from core.paths import user_data_root
            guard = desktop_instance(user_data_root() / 'desktop.lock', url)
        with guard as existing:
            if existing is not None:
                address = existing.get('url')
                print(f'TouchSign 已在运行：{address or "正在启动，请稍后重试"}，不再启动第二个采集进程。')
                if address and not args.no_show:
                    from gui.browser_launcher import open_browser
                    open_browser(address)
                return
            from gui.runtime import prepare_runtime
            prepare_runtime()
            from gui.app import run_app
            run_app(config, native=False, show=not args.no_show, port=port)


def main(argv=None):
    parser = argparse.ArgumentParser(description='触象 · TouchSign 中文识别界面')
    parser.add_argument('--config')
    parser.add_argument('--self-test-report', help='Exercise the actual embedded GUI and exit with a JSON report')
    parser.add_argument('--diagnostics-report', help='Offline model, speech, SDK and robot-cycle check')
    parser.add_argument('--no-show', action='store_true', help='Ubuntu：仅启动本地服务，不打开浏览器')
    parser.add_argument('--port', type=int, help='Ubuntu：指定本地服务端口')
    args = parser.parse_args(argv)
    from core.paths import user_data_root
    root = user_data_root()
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    if getattr(sys, 'frozen', False):
        output = (logs / 'desktop-startup.log').open('a', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr = output
        temp = root / 'temp'
        temp.mkdir(parents=True, exist_ok=True)
        os.environ['TEMP'] = os.environ['TMP'] = str(temp)
        tempfile.tempdir = str(temp)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        handlers=[logging.FileHandler(logs / 'desktop.log', encoding='utf-8')])
    try:
        from core.windows_diagnostics import token_integrity
        logging.info('Process integrity=%s executable=%s bundle=%s', token_integrity(), sys.executable, getattr(sys, '_MEIPASS', None))
        if args.diagnostics_report:
            from robot.self_test import run_self_test
            return run_self_test(args.diagnostics_report)
        from core.config import load_config, PROJECT_ROOT
        default_config = ('ubuntu/config.yaml' if sys.platform == 'linux' else
                          'config.yaml' if getattr(sys, 'frozen', False) else 'raw/config.yaml')
        config_path = Path(args.config) if args.config else PROJECT_ROOT / 'config' / default_config
        config = load_config(config_path)
        _run_ui(config, args)
        if args.self_test_report:
            report = json.loads(Path(args.self_test_report).read_text(encoding='utf-8'))
            return 0 if report.get('passed', report.get('offline_checks_passed', False)) else 1
        return 0
    except KeyboardInterrupt:
        # Uvicorn re-raises SIGINT after its shutdown hooks have completed.
        print('TouchSign 已退出。')
        return 0
    except Exception as exc:
        logging.exception('Native desktop startup failed')
        report_path = args.self_test_report or args.diagnostics_report
        if report_path:
            Path(report_path).write_text(json.dumps({'passed': False, 'error': repr(exc),
                'traceback': traceback.format_exc()}, ensure_ascii=False, indent=2), encoding='utf-8')
        elif sys.platform == 'win32':
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,
                f'TouchSign 启动失败：{exc}\n\n诊断日志：{logs / "desktop.log"}', 'TouchSign', 0x10)
        else:
            print(f'TouchSign 启动失败：{exc}\n诊断日志：{logs / "desktop.log"}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
