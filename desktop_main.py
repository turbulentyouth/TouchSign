"""Native TouchSign entry point: no HTTP server, browser or Python.NET."""
from pathlib import Path
import argparse
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import traceback


def main():
    parser = argparse.ArgumentParser(description='TouchSign native desktop application')
    parser.add_argument('--config')
    parser.add_argument('--self-test-report', help='Exercise the actual embedded GUI and exit with a JSON report')
    parser.add_argument('--diagnostics-report', help='Offline model, speech, SDK and robot-cycle check')
    args = parser.parse_args()
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
        config_path = Path(args.config) if args.config else PROJECT_ROOT / 'config' / ('config.yaml' if getattr(sys, 'frozen', False) else 'raw/config.yaml')
        config = load_config(config_path)
        from gui.embedded import run_desktop
        run_desktop(config, self_test_report=args.self_test_report)
        if args.self_test_report:
            report = json.loads(Path(args.self_test_report).read_text(encoding='utf-8'))
            return 0 if report.get('passed', report.get('offline_checks_passed', False)) else 1
        return 0
    except Exception as exc:
        logging.exception('Native desktop startup failed')
        report_path = args.self_test_report or args.diagnostics_report
        if report_path:
            Path(report_path).write_text(json.dumps({'passed': False, 'error': repr(exc),
                'traceback': traceback.format_exc()}, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,
                f'TouchSign 启动失败：{exc}\n\n诊断日志：{logs / "desktop.log"}', 'TouchSign', 0x10)
        return 1


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
