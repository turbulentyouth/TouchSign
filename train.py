"""Train YOLO on an 80/20 grouped split, then export and validate ONNX rejection.

Examples:
  python train.py --data data/raw --epochs 80 --device 0 --deploy
  python train.py --input raw --data data/raw --epochs 80 --device 0
  python train.py --data data/raw --prepare-only
  python train.py --smoke  # isolated synthetic engineering check; never deploys
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='视触觉生肖 YOLO 分类训练、ONNX 导出与未知物体拒识校准')
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'raw', help='collect.py 生成的采集目录')
    parser.add_argument('--input', choices=('diff', 'raw'), default='diff',
                        help='训练和验证使用的图像：diff 为差分图（默认），raw 为传感器原图')
    parser.add_argument('--output', type=Path, help='新的训练输出目录（必须为空）')
    parser.add_argument('--model', default='yolo11n-cls.pt', help='YOLO 分类预训练权重；离线从零训练可用 yolo11n-cls.yaml')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='0', help='GPU 序号，或 cpu')
    parser.add_argument('--workers', type=int, default=0, help='Windows 默认 0，避免子进程与受限系统冲突')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-per-class', type=int, default=1, help='可独立划分的类别在训练、验证两侧各需的最少图像数')
    parser.add_argument('--min-recall', type=float, default=0.8, help='每个生肖最低正确接受率，低于此值不发布')
    parser.add_argument('--max-far', type=float, default=0.0, help='留出未知/无接触负例允许误接收比例，默认零')
    parser.add_argument('--prepare-only', action='store_true', help='校验数据，按分组划分约 80%% 训练、20%% 验证')
    parser.add_argument('--export-only', type=Path, help='导出使用同一数据划分训练的权重，并在验证集上校准')
    parser.add_argument('--deploy', action='store_true', help='验证集覆盖全部类别且指标达标后接入 GUI')
    parser.add_argument('--smoke', action='store_true', help='合成数据单轮工程烟测，禁止接入 GUI')
    args = parser.parse_args(argv)
    if args.image_size < 32 or args.image_size > 1024 or args.image_size % 32:
        parser.error('--image-size 必须是 32 的倍数，范围 32..1024')
    if min(args.epochs, args.batch, args.min_per_class) < 1 or args.workers < 0:
        parser.error('epochs、batch、min-per-class 必须为正数，workers 不能为负数')
    if not 0 < args.min_recall <= 1 or not 0 <= args.max_far < 1:
        parser.error('min-recall 必须在 (0,1]，max-far 必须在 [0,1)')
    if args.smoke and (args.deploy or args.export_only):
        parser.error('合成烟测不能发布到 GUI，也不能与 export-only 混用')
    return args


def configure_environment():
    config_dir = ROOT / '.cache' / 'ultralytics'
    (config_dir / 'Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(config_dir))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    from gui.runtime import prepare_runtime
    prepare_runtime()


def synthetic_smoke_data(root: Path, seed: int):
    """Artificial data exercises code only; it cannot establish sensor accuracy."""
    import cv2
    import numpy as np
    from core.inference import CLASS_NAMES, INPUT_KIND
    rng = np.random.default_rng(seed)
    for session in range(5):
        directory = root / f'smoke_session_{session}'
        directory.mkdir(parents=True, exist_ok=True)
        baseline = np.full((72, 96, 3), 25, dtype=np.uint8)
        baseline_path = directory / 'baseline.png'
        cv2.imencode('.png', baseline)[1].tofile(baseline_path)
        records = []
        for class_id, label in enumerate(CLASS_NAMES):
            target = directory / label
            target.mkdir(exist_ok=True)
            for i in range(2):
                frame = rng.integers(0, 15, (72, 96, 3), dtype=np.uint8)
                cv2.circle(frame, (12 + class_id * 5, 20 + i * 25), 8 + session,
                           (20 + class_id * 13, 120, 220), -1)
                path = target / f'{i}.png'
                cv2.imencode('.png', frame)[1].tofile(path)
                raw_path = target / f'{i}-raw.png'
                cv2.imencode('.png', cv2.add(baseline, frame))[1].tofile(raw_path)
                records.append({'image': path.relative_to(root).as_posix(), 'label': label,
                                'raw_image': raw_path.relative_to(root).as_posix(),
                                'baseline': baseline_path.relative_to(root).as_posix(),
                                'session_id': directory.name, 'episode_id': f'{label}-{i}',
                                'synthetic': True, 'input_kind': INPUT_KIND,
                                'contact_detected': label != 'no_contact'})
        (directory / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')


def export_onnx(weights: Path, destination: Path, image_size: int, parity_samples: list[dict]):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    from core.inference import CLASS_NAMES, preprocess
    from training.data import read_image
    from training.yolo import EmbeddingClassifier

    loaded = YOLO(str(weights), task='classify')
    if [loaded.names[i] for i in range(len(loaded.names))] != list(CLASS_NAMES):
        raise ValueError('训练模型类别不匹配，需要 12 生肖 + unknown + no_contact（字母排序）')
    original = loaded.model.cpu().float().eval()
    wrapper = EmbeddingClassifier(original).eval()
    dummy = torch.from_numpy(preprocess(read_image(parity_samples[0]['path']), image_size))
    with torch.inference_mode():
        original_probs = original(dummy)[0]
        exported_probs, _ = wrapper(dummy)
        np.testing.assert_allclose(original_probs.numpy(), exported_probs.numpy(), atol=1e-5, rtol=1e-4)
    torch.onnx.export(wrapper, dummy, str(destination), export_params=True, opset_version=17,
                      do_constant_folding=True, input_names=['images'],
                      output_names=['probabilities', 'embedding'], dynamo=False)
    onnx.checker.check_model(str(destination), full_check=True)
    session = ort.InferenceSession(str(destination), providers=['CPUExecutionProvider'])
    max_error = 0.0
    for sample in parity_samples[:5]:
        tensor = preprocess(read_image(sample['path']), image_size)
        with torch.inference_mode():
            expected = wrapper(torch.from_numpy(tensor))
        actual = session.run(None, {'images': tensor})
        for wanted, got in zip(expected, actual):
            np.testing.assert_allclose(wanted.numpy(), got, atol=2e-5, rtol=2e-4)
            max_error = max(max_error, float(np.max(np.abs(wanted.numpy() - got))))
    return session, {'onnx_checked': True, 'samples': min(5, len(parity_samples)),
                     'max_abs_error': max_error, 'size_mb': destination.stat().st_size / 1024 ** 2}


def extract_features(session, samples: list[dict], image_size: int):
    from core.inference import preprocess
    from training.data import read_image
    features = []
    for sample in samples:
        probabilities, embedding = session.run(['probabilities', 'embedding'],
                                               {'images': preprocess(read_image(sample['path']), image_size)})
        features.append({'label': sample['label'], 'probabilities': probabilities[0], 'embedding': embedding[0]})
    return features


def deploy(model_path: Path, manifest_path: Path):
    """Replace a matching pair only after validation; keep the previous pair."""
    from core.inference import OnnxClassifier
    OnnxClassifier(model_path, manifest_path)  # hash, shape, calibration and runtime load checks
    destination = ROOT / 'models'
    destination.mkdir(parents=True, exist_ok=True)
    old = [destination / 'best.onnx', destination / 'best.json']
    if any(path.exists() for path in old):
        archive = destination / 'archive' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        archive.mkdir(parents=True)
        for path in old:
            if path.is_file():
                shutil.copy2(path, archive / path.name)
    shutil.copy2(model_path, destination / 'best.onnx.tmp')
    shutil.copy2(manifest_path, destination / 'best.json.tmp')
    os.replace(destination / 'best.onnx.tmp', destination / 'best.onnx')
    os.replace(destination / 'best.json.tmp', destination / 'best.json')


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_environment()
    from training.data import prepare_dataset
    from core.inference import CLASS_NAMES, INPUT_KIND, RAW_INPUT_KIND, PREPROCESS_VERSION, file_sha256
    input_kind = RAW_INPUT_KIND if args.input == 'raw' else INPUT_KIND
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_name = f'raw_{stamp}' if args.input == 'raw' else stamp
    output = (args.output or ROOT / ('.cache/model-smoke' if args.smoke else 'runs') / run_name).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'训练输出目录非空，请指定新目录：{output}')
    if args.smoke:
        args.data = output / 'synthetic_raw'
        synthetic_smoke_data(args.data, args.seed)
        args.model, args.epochs, args.image_size = 'yolo11n-cls.yaml', 1, 64
        args.device, args.batch, args.min_per_class = 'cpu', 14, 1
    if args.export_only:
        weights = args.export_only.resolve()
        if not weights.is_file() or weights.stat().st_size == 0:
            raise ValueError(f'训练权重不存在或为空：{weights}')
        if weights.parent.name == 'weights' and weights.parent.parent.name == 'fit':
            split_path = weights.parents[2] / 'dataset' / 'split.json'
            if split_path.is_file():
                recorded_kind = json.loads(split_path.read_text(encoding='utf-8')).get('input_kind', INPUT_KIND)
                if recorded_kind != input_kind:
                    raise ValueError('导出时的 --input 必须与该权重训练时的图像类型一致')
    split, split_report = prepare_dataset(args.data, output / 'dataset', seed=args.seed,
                                          min_per_class=args.min_per_class, input_kind=input_kind)
    print(json.dumps(split_report, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        print(f'数据分割完成：{output / "dataset"}')
        return 0
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from training.yolo import TactileTrainer
    from training.calibrate import calibrate, passes_release
    settings.update({'sync': False, 'wandb': False, 'mlflow': False, 'tensorboard': False})
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise ValueError('当前 CUDA 不可用；可使用 --device cpu 或安装匹配显卡的训练环境')
    if not args.export_only:
        yolo = YOLO(args.model, task='classify')
        yolo.train(trainer=TactileTrainer, data=str(output / 'dataset'), epochs=args.epochs,
                   imgsz=args.image_size, batch=args.batch, device=args.device, workers=args.workers,
                   seed=args.seed, deterministic=True, project=str(output), name='fit', exist_ok=False,
                   patience=15, optimizer='AdamW', lr0=0.001, pretrained=not args.model.endswith('.yaml'),
                   plots=False, cache=False, save=True, val=True, verbose=False,
                   auto_augment=None, erasing=0, fliplr=0, flipud=0)
        weights = Path(yolo.trainer.best)
    model_path = output / 'best.onnx'
    session, parity = export_onnx(weights, model_path, args.image_size, split['val'])
    manifest = {'schema_version': 1, 'status': 'unvalidated', 'synthetic': split_report['synthetic'],
                'model': 'YOLO-classify-with-embedding', 'class_names': list(CLASS_NAMES),
                'preprocess': PREPROCESS_VERSION, 'input_kind': input_kind, 'image_size': args.image_size,
                'sha256': file_sha256(model_path), 'created_utc': datetime.now(timezone.utc).isoformat(),
                'versions': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
                'split_report': split_report, 'onnx_parity': parity,
                'release_requirements': {'min_per_class_recall': args.min_recall, 'max_far': args.max_far},
                'evaluation_scope': 'validation_only',
                'limitation': 'The same validation split selects weights and rejection thresholds and supplies the reported metrics.'}
    train_features = extract_features(session, split['train'], args.image_size)
    val_features = extract_features(session, split['val'], args.image_size)
    manifest, validation = calibrate(train_features, val_features, manifest, max_far=args.max_far)
    manifest['validation'] = validation
    release = (not manifest['synthetic']
               and passes_release(validation, min_recall=args.min_recall, max_far=args.max_far))
    if release:
        manifest['status'] = 'validated'
    manifest_path = model_path.with_suffix('.json')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'output': str(output), 'status': manifest['status'], 'synthetic': manifest['synthetic'],
              'input_kind': input_kind,
              'evaluation_scope': manifest['evaluation_scope'], 'split_report': split_report,
              'onnx_parity': parity, 'validation': validation,
              'deployed': bool(args.deploy and release)}
    if validation['missing_classes']:
        print('验证集缺少独立样本的类别：' + ', '.join(validation['missing_classes']) + '。模型已导出，可使用 GUI 预览模式。')
    if args.deploy and release:
        deploy(model_path, manifest_path)
        print('已发布 models/best.onnx 与 best.json，GUI 可重新加载模型。')
    elif args.deploy:
        print('未通过发布门槛，保留当前 GUI 权重。请查看 report.json 并补采独立真实数据。')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'weights': str(weights), 'status': manifest['status'],
                      'onnx_parity': parity, 'validation_far': validation['false_accept_rate'],
                      'validation_correct_accept_rate': validation['correct_accept_rate']}, ensure_ascii=False, indent=2))
    return 2 if args.deploy and not release else 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f'[TouchSign 训练] {exc}', file=sys.stderr)
        raise SystemExit(2) from None
