"""YOLO-specific implementation; imported only by the offline training command."""
from __future__ import annotations

from copy import copy

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

from core.inference import preprocess


class TactileTransform:
    """Preserve the whole contact imprint; no default classification center crop."""
    def __init__(self, size: int, augment: bool = False):
        self.size = size
        self.augment = augment

    def __call__(self, image: Image.Image) -> torch.Tensor:
        bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        if self.augment:
            # Small photometric variation only: physical rotation/pressure diversity
            # comes from real independent contacts instead of implausible crops.
            gain = float(torch.empty(1).uniform_(0.9, 1.1))
            bgr = np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return torch.from_numpy(preprocess(bgr, self.size)[0])


class TactileDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=''):
        super().__init__(root, args, augment, prefix)
        self.torch_transforms = TactileTransform(args.imgsz, augment)


class TactileValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        return TactileDataset(img_path, self.args, augment=False, prefix=self.args.split)


class TactileTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return TactileDataset(img_path, self.args, augment=mode == 'train', prefix=mode)

    def get_validator(self):
        self.loss_names = ['loss']
        return TactileValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


class EmbeddingClassifier(torch.nn.Module):
    """Expose YOLO Classify's pooled feature, normalized for cosine rejection."""
    def __init__(self, classification_model):
        super().__init__()
        layers = list(classification_model.model.children())
        if any(getattr(layer, 'f', -1) != -1 for layer in layers):
            raise ValueError('只支持顺序结构 YOLO 分类骨干，请使用 yolo11n-cls')
        self.backbone = torch.nn.Sequential(*layers[:-1])
        head = layers[-1]
        if not all(hasattr(head, name) for name in ('conv', 'pool', 'linear')):
            raise ValueError('模型没有受支持的 YOLO Classify head')
        self.conv, self.pool, self.linear = head.conv, head.pool, head.linear

    def forward(self, image):
        feature = self.pool(self.conv(self.backbone(image))).flatten(1)
        probabilities = self.linear(feature).softmax(dim=1)
        embedding = torch.nn.functional.normalize(feature, dim=1)
        return probabilities, embedding
