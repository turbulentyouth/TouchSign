"""Small CPU-only ONNX classifier with calibrated open-set rejection.

This module deliberately never imports torch, torchvision or ultralytics. Models
must have been exported and calibrated by train.py; an arbitrary ONNX classifier
cannot silently become a zodiac recognizer.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.zodiac import ZODIAC_KEYS

CLASS_NAMES = tuple(sorted((*ZODIAC_KEYS, 'unknown', 'no_contact')))
PREPROCESS_VERSION = 'letterbox-rgb-255-v1'
INPUT_KIND = 'absdiff_bgr_v1'
RAW_INPUT_KIND = 'raw_bgr_v1'


def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """Full-frame letterbox; identical code is used by training transforms."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('需要非空 uint8 BGR 三通道图像')
    if frame.dtype != np.uint8 or not frame.size:
        raise ValueError('需要非空 uint8 BGR 三通道图像')
    h, w = frame.shape[:2]
    ratio = size / max(h, w)
    width, height = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - height) // 2, (size - width) // 2
    canvas[top:top + height, left:left + width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    key: str | None
    predicted_key: str
    confidence: float
    margin: float
    distance: float
    accepted: bool
    reason: str
    candidates: tuple[tuple[str, float], ...]
    latency_ms: float


def decide(probabilities: np.ndarray, embedding: np.ndarray, manifest: dict,
           *, latency_ms: float = 0.0, preview: bool = False) -> ModelPrediction:
    """Rank all classes and apply the calibrated rejection policy outside preview."""
    names = manifest['class_names']
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if (len(probabilities) != len(names) or not np.all(np.isfinite(probabilities))
            or not np.all(np.isfinite(embedding)) or np.linalg.norm(embedding) < 1e-8
            or np.min(probabilities) < -1e-5 or np.max(probabilities) > 1.00001
            or abs(float(probabilities.sum()) - 1) > 0.01):
        raise ValueError('模型输出无效，已停止识别')
    order = np.argsort(probabilities)[::-1]
    index = int(order[0])
    predicted = names[index]
    confidence = float(probabilities[index])
    margin = confidence - float(probabilities[int(order[1])])
    candidates = tuple((names[int(i)], float(probabilities[int(i)])) for i in order[:3])
    distance = 1.0
    reason = 'unknown' if predicted == 'unknown' else 'no_contact'
    if predicted in ZODIAC_KEYS:
        prototype = np.asarray(manifest['prototypes'][predicted], dtype=np.float32)
        if prototype.shape != embedding.shape:
            raise ValueError('模型特征维度与校准文件不匹配')
        norm = float(np.linalg.norm(embedding) * np.linalg.norm(prototype))
        distance = max(0.0, 1.0 - float(np.dot(embedding, prototype)) / max(norm, 1e-8))
        if preview:
            reason = 'preview_top1'
        else:
            limits = manifest['thresholds'][predicted]
            reason = ('low_confidence' if confidence < limits['confidence'] else
                      'ambiguous' if margin < limits['margin'] else
                      'out_of_distribution' if distance > limits['distance'] else 'accepted')
    accepted = reason in ('accepted', 'preview_top1')
    return ModelPrediction(predicted if accepted else None, predicted, confidence, margin,
                           distance, accepted, reason, candidates, latency_ms)


class OnnxClassifier:
    def __init__(self, model_path: str | Path, manifest_path: str | Path | None = None,
                 *, threads: int = 2, preview: bool = False) -> None:
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path) if manifest_path else self.model_path.with_suffix('.json')
        self.preview = preview
        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            raise FileNotFoundError(f'未安装经过训练的 ONNX 权重：{self.model_path}')
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f'缺少模型校准文件：{self.manifest_path}')
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        self._validate_manifest()
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)':
            raise ValueError('ONNX 必须只有一个 float32 图像输入')
        self.input_name = inputs[0].name
        if inputs[0].shape != [1, 3, self.size, self.size]:
            raise ValueError('ONNX 输入尺寸与校准文件不一致')
        outputs = self.session.get_outputs()
        if {item.name for item in outputs} != {'probabilities', 'embedding'}:
            raise ValueError('ONNX 必须包含 probabilities 和 embedding 输出')
        shapes = {item.name: item.shape for item in outputs}
        dimension = len(self.manifest['prototypes'][ZODIAC_KEYS[0]])
        if shapes['probabilities'] != [1, 14] or shapes['embedding'] != [1, dimension]:
            raise ValueError('ONNX 输出尺寸与校准文件不一致')

    def _validate_manifest(self) -> None:
        m = self.manifest
        if (m.get('schema_version') != 1 or m.get('preprocess') != PREPROCESS_VERSION
                or m.get('input_kind') not in (INPUT_KIND, RAW_INPUT_KIND)):
            raise ValueError('不支持的模型格式或预处理方法，请通过 train.py 重新导出')
        self.input_kind = m['input_kind']
        if m.get('class_names') != list(CLASS_NAMES):
            raise ValueError('模型类别必须包含十二生肖、unknown、no_contact，且按实际训练顺序保存')
        preview_export = (self.preview and m.get('status') == 'unvalidated'
                          and m.get('synthetic') is False)
        if m.get('synthetic', False) or (m.get('status') != 'validated' and not preview_export):
            raise ValueError('模型尚未通过真实数据校准与留出集验证，禁止用于实际识别')
        if m.get('sha256') != file_sha256(self.model_path):
            raise ValueError('ONNX 与校准文件不匹配（SHA256 校验失败）')
        self.size = int(m.get('image_size', 0))
        if self.size < 32 or self.size > 1024 or self.size % 32:
            raise ValueError('模型输入尺寸无效')
        dimensions = set()
        for key in ZODIAC_KEYS:
            prototype = np.asarray(m.get('prototypes', {}).get(key, []), dtype=np.float32)
            if (prototype.ndim != 1 or not prototype.size or not np.all(np.isfinite(prototype))
                    or np.linalg.norm(prototype) < 1e-8):
                raise ValueError(f'缺少有效的 {key} 特征原型')
            dimensions.add(prototype.size)
            limits = m.get('thresholds', {}).get(key, {})
            for field in ('confidence', 'margin', 'distance'):
                value = limits.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'{key} 的 {field} 拒识阈值无效')
        if len(dimensions) != 1:
            raise ValueError('特征原型维度不一致')

    def predict(self, frame: np.ndarray) -> ModelPrediction:
        """Classify the BGR input specified by the manifest after the live contact gate."""
        started = time.perf_counter()
        tensor = preprocess(frame, self.size)
        probabilities, embedding = self.session.run(['probabilities', 'embedding'], {self.input_name: tensor})
        return decide(probabilities, embedding, self.manifest, preview=self.preview,
                      latency_ms=(time.perf_counter() - started) * 1000)
