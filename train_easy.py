"""Double-click incremental training: replay, fixed holdouts, compare, atomic switch."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def predict_rows(model, rows):
    from training.data import read_image
    predictions = {}
    for row in rows:
        prediction = model.predict(read_image(row['path']))
        predictions[row['sha256']] = {'label': prediction.predicted_key,
                                       'confidence': prediction.confidence}
    return predictions


def check_retention(rows, before, after):
    """Protect every previously correct decision and its confidence, not just average accuracy."""
    regressions = []
    for row in rows:
        key = row['sha256']
        old, new = before[key], after[key]
        if old['label'] == row['label'] and (
            new['label'] != row['label'] or
            (old['confidence'] >= .5 and new['confidence'] < .5) or
            new['confidence'] < old['confidence'] - .15
        ):
            regressions.append({'sha256': key, 'label': row['label'], 'before': old, 'after': new})
    return regressions


def run_training(root=ROOT):
    from core.inference import OnnxClassifier, RAW_INPUT_KIND, file_sha256
    from core.portable_model import FACTORY, activate, local_path, selected
    from training.data import read_samples
    from training.portable import load_snapshot, merge_new, save_snapshot

    current = selected(root)
    base = load_snapshot(local_path(root, current['snapshot']))
    factory = load_snapshot(root / 'data/seed')
    external = load_snapshot(root / 'data/regression')
    batches = []
    for name, grouping in (('manual', 'session'), ('corrections', 'frame')):
        source = root / 'data' / name
        if list(source.glob('*/samples.jsonl')):
            batches.extend(read_samples(source, input_kind=RAW_INPUT_KIND, manual_grouping=grouping))
    rows, added = merge_new(base, batches, external)
    if not added:
        print('没有未训练的新数据。请先使用人工采集，或在识别界面保存纠错样本。当前模型保持不变。', flush=True)
        return 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run = root / 'runs' / f'easy_{stamp}'
    run.mkdir(parents=True)
    snapshot, output = run / 'source', run / 'training'
    print(f'旧数据 {len(base)} 张，新增 {len(added)} 张。正在保存本轮数据快照…', flush=True)
    save_snapshot(rows, snapshot)
    print('开始小步续训（CPU，最多 16 轮）。窗口会显示进度，请等待完成。', flush=True)
    command = [sys.executable, '-u', str(root / 'train.py'), '--input', 'raw',
               '--data', str(snapshot), '--output', str(output),
               '--model', str(local_path(root, current['model_dir']) / 'best.pt'),
               '--reference-split', str(local_path(root, current['reference'])),
               '--allow-missing-unknown', '--epochs', '16', '--learning-rate', '0.00005',
               '--warmup-bias-lr', '0', '--device', 'cpu', '--workers', '0', '--batch', '16', '--no-amp']
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', YOLO_AUTOINSTALL='false',
               YOLO_OFFLINE='true', WANDB_MODE='disabled', OMP_NUM_THREADS=str(min(8, os.cpu_count() or 1)))
    with (run / 'training.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as child:
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
    if code:
        raise RuntimeError(f'训练未完成（代码 {code}），当前模型未改变。日志：{run / "training.log"}')
    shutil.copy2(output / 'fit/weights/best.pt', output / 'best.pt')

    print('正在逐张比较新旧模型；检查旧数据、固定验证集及新增样本…', flush=True)
    candidate = OnnxClassifier(output / 'best.onnx', preview=True)
    old = OnnxClassifier(local_path(root, current['model_dir']) / 'best.onnx', preview=True)
    original = OnnxClassifier(root / FACTORY['model_dir'] / 'best.onnx', preview=True)
    evaluation = [*rows, *external]
    before, after = predict_rows(old, evaluation), predict_rows(candidate, evaluation)
    original_predictions = predict_rows(original, [*factory, *external])
    regressions = check_retention(evaluation, before, after)
    factory_regressions = check_retention([*factory, *external], original_predictions, after)
    def correct(predictions):
        return sum(predictions[r['sha256']]['label'] == r['label'] and
                   predictions[r['sha256']]['confidence'] >= .5 for r in added)
    new_before, new_after = correct(before), correct(after)
    improved = new_after > new_before
    passed = not regressions and not factory_regressions and improved
    split = json.loads((output / 'dataset/split.json').read_text(encoding='utf-8'))
    report = {'activated': passed, 'new_samples': len(added), 'new_labels': dict(Counter(r['label'] for r in added)),
              'new_correct_before': new_before, 'new_correct_after': new_after,
              'regressions': regressions, 'factory_regressions': factory_regressions,
              'split': split, 'candidate_sha256': file_sha256(output / 'best.onnx'),
              'evaluation': [{'sha256': r['sha256'], 'label': r['label'],
                              'before': before[r['sha256']], 'after': after[r['sha256']]} for r in evaluation],
              'limitation': '新增训练样本的正确率是拟合检查。验证集参与选权重；本包保持预览模式，不代表独立泛化验证。'}
    (run / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if passed:
        record = {'model_dir': output.relative_to(root).as_posix(),
                  'snapshot': snapshot.relative_to(root).as_posix(),
                  'reference': (output / 'dataset').relative_to(root).as_posix(),
                  'weights_sha256': file_sha256(output / 'best.pt')}
        activate(root, record, current)
        print('训练完成：旧识别检查通过，新增样本有改善。下次打开识别将自动使用新模型。', flush=True)
    else:
        reason = '出现旧样本回退' if regressions or factory_regressions else '新增样本未出现正确率提升'
        print(f'训练完成，但{reason}，继续使用原模型。本轮结果已保存。', flush=True)
    print(f'新增样本（含训练样本）正确数：{new_before} → {new_after} / {len(added)}。报告：{run / "comparison.json"}', flush=True)
    return 0


def main():
    from core.instance import desktop_instance
    from core.paths import user_data_root
    with desktop_instance(user_data_root() / 'desktop.lock', '') as existing:
        if existing is not None:
            raise RuntimeError('请先退出采集、识别或另一训练窗口，再启动训练。')
        return run_training()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'训练停止：{exc or "已取消"}。当前模型保持不变。', file=sys.stderr)
        raise SystemExit(1)
