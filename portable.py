"""Entry point used by the copy-and-run Windows bundle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def prepare():
    os.chdir(ROOT)
    os.environ['TOUCHSIGN_PORTABLE'] = '1'
    os.environ['PYTHONUTF8'] = '1'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['MPLBACKEND'] = 'Agg'
    config_dir = ROOT / '.cache/ultralytics'
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    os.environ.setdefault('OMP_NUM_THREADS', str(min(8, os.cpu_count() or 1)))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    (ROOT / 'logs').mkdir(exist_ok=True)
    for name in ('manual', 'corrections'):
        (ROOT / 'data' / name).mkdir(parents=True, exist_ok=True)


def self_check(root=ROOT):
    import ssl
    import cv2
    import numpy as np
    import onnx
    import torch
    import torchvision
    from gui.runtime import prepare_runtime
    prepare_runtime()
    import nicegui
    from ultralytics import YOLO
    from core.inference import OnnxClassifier, preprocess
    from core.portable_model import selected, local_path
    from training.data import read_image
    from training.portable import load_snapshot
    from training.yolo import EmbeddingClassifier
    from core.instance import desktop_instance
    from gui.collection_canvas import _chinese_font
    print('正在检查全部训练图像、模型和依赖…', flush=True)
    record = selected(root)
    rows = load_snapshot(local_path(root, record['snapshot']))
    external = load_snapshot(root / 'data/regression')
    directory = local_path(root, record['model_dir'])
    model = OnnxClassifier(directory / 'best.onnx', preview=True)
    frame = read_image(rows[0]['path'])
    prediction = model.predict(frame)
    loaded = YOLO(str(directory / 'best.pt'), task='classify')
    wrapper = EmbeddingClassifier(loaded.model.cpu().float().eval()).eval()
    with torch.inference_mode():
        expected = wrapper(torch.from_numpy(preprocess(frame, model.size)))
    actual = model.session.run(None, {model.input_name: preprocess(frame, model.size)})
    for wanted, got in zip(expected, actual):
        np.testing.assert_allclose(wanted.numpy(), got, atol=2e-5, rtol=2e-4)
    lock = root / 'logs/self-check.lock'
    with desktop_instance(lock, 'self-check') as owner:
        assert owner is None
        with desktop_instance(lock, 'duplicate') as duplicate:
            assert duplicate is not None and duplicate.get('url') == 'self-check'
    with desktop_instance(lock, 'reopened') as owner:
        assert owner is None
    font = str(_chinese_font(None))
    report = {'passed': True, 'platform': sys.platform, 'python': sys.version,
              'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'cv2': cv2.__version__, 'nicegui': nicegui.__version__, 'ssl': ssl.OPENSSL_VERSION,
              'data_samples': len(rows), 'external_regression': len(external), 'font': font,
              'model_sha256': model.manifest['sha256'], 'example_prediction': prediction.predicted_key,
              'onnx_pytorch_parity': True, 'process_lock': True,
              'hardware_tested': False}
    (root / 'logs/self-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('自检通过。相机连接和画面需在采集入口确认。报告：logs/self-check.json', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='TouchSign Windows 便携版')
    parser.add_argument('action', choices=('collect', 'recognize', 'train', 'rollback', 'check'))
    args = parser.parse_args(argv)
    prepare()
    if args.action == 'collect':
        from collect_manual import main as collect
        return collect(['--config', str(ROOT / 'config/windows/config.yaml')])
    if args.action == 'train':
        from train_easy import main as train
        return train()
    if args.action == 'recognize':
        import yaml
        from core.portable_model import local_path, selected
        record = selected(ROOT)
        config = yaml.safe_load((ROOT / 'config/windows/config.yaml').read_text(encoding='utf-8'))
        model = local_path(ROOT, record['model_dir'])
        config['recognition'].update(model_path=str(model / 'best.onnx'), manifest_path=str(model / 'best.json'))
        target = ROOT / 'data/app/recognition.yaml'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        from desktop_main import main as recognize
        return recognize(['--manual', '--config', str(target)])
    if args.action == 'rollback':
        from core.instance import desktop_instance
        from core.paths import user_data_root
        from core.portable_model import rollback
        with desktop_instance(user_data_root() / 'desktop.lock', '') as existing:
            if existing is not None:
                raise RuntimeError('请先关闭采集、训练和识别窗口，再回退模型。')
            print('已切换到上一个模型。' if rollback(ROOT) else '当前是随包模型，无需回退。')
        return 0
    return self_check()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('已退出。')
        raise SystemExit(130)
    except Exception as exc:
        (ROOT / 'logs').mkdir(exist_ok=True)
        (ROOT / 'logs/last-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(f'启动失败：{exc}\n详细日志：{ROOT / "logs/last-error.txt"}', file=sys.stderr)
        raise SystemExit(1)
