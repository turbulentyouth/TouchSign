"""Fit rejection prototypes on training data and calibrate thresholds on validation data."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from core.inference import CLASS_NAMES, decide
from core.zodiac import ZODIAC_KEYS


def fit_prototypes(train: list[dict]) -> tuple[dict, dict]:
    prototypes, distances = {}, {}
    for key in ZODIAC_KEYS:
        features = np.stack([sample['embedding'] for sample in train if sample['label'] == key]).astype(np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
        prototype = features.mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
        prototypes[key] = prototype.tolist()
        distances[key] = np.maximum(0, 1 - features @ prototype)
    return prototypes, distances


def evaluate(samples: list[dict], manifest: dict) -> dict:
    classes = {key: {'count': 0, 'correct_accepted': 0, 'wrong_accepted': 0, 'rejected': 0} for key in CLASS_NAMES}
    confusion = {key: {} for key in CLASS_NAMES}
    for sample in samples:
        result = decide(sample['probabilities'], sample['embedding'], manifest)
        row = classes[sample['label']]
        row['count'] += 1
        predicted = result.key if result.accepted else 'rejected'
        confusion[sample['label']][predicted] = confusion[sample['label']].get(predicted, 0) + 1
        row['rejected' if not result.accepted else 'correct_accepted' if result.key == sample['label'] else 'wrong_accepted'] += 1
    negatives = [classes[key] for key in ('unknown', 'no_contact')]
    positives = [classes[key] for key in ZODIAC_KEYS]
    negative_count = sum(row['count'] for row in negatives)
    false_accepts = sum(row['wrong_accepted'] for row in negatives)
    positive_count = sum(row['count'] for row in positives)
    correct = sum(row['correct_accepted'] for row in positives)
    wrong = sum(row['wrong_accepted'] for row in positives)
    per_class = {key: classes[key]['correct_accepted'] / max(1, classes[key]['count']) for key in ZODIAC_KEYS}
    missing_classes = [key for key in CLASS_NAMES if classes[key]['count'] == 0]
    return {'counts': classes, 'missing_classes': missing_classes, 'confusion': confusion, 'negative_count': negative_count,
            'false_accepts': false_accepts, 'false_accept_rate': false_accepts / max(1, negative_count),
            'positive_count': positive_count, 'correct_accept_rate': correct / max(1, positive_count),
            'wrong_zodiac_rate': wrong / max(1, positive_count),
            'positive_rejection_rate': sum(row['rejected'] for row in positives) / max(1, positive_count),
            'per_class_correct_accept_rate': per_class}


def calibrate(train: list[dict], validation: list[dict], manifest: dict, *, max_far: float = 0.0) -> tuple[dict, dict]:
    prototypes, distances = fit_prototypes(train)
    base = deepcopy(manifest)
    base['prototypes'] = prototypes
    best, best_report, best_score = None, None, (-1.0, -1.0)
    for confidence in (0.85, 0.90, 0.95, 0.98, 0.995):
        for margin in (0.15, 0.25, 0.4):
            for quantile in (0.9, 0.95, 0.99):
                candidate = deepcopy(base)
                candidate['thresholds'] = {
                    key: {'confidence': confidence, 'margin': margin,
                          'distance': min(0.35, max(0.005, float(np.quantile(distances[key], quantile)) * 1.2))}
                    for key in ZODIAC_KEYS}
                report = evaluate(validation, candidate)
                if report['false_accept_rate'] > max_far or report['wrong_zodiac_rate'] > 0:
                    continue
                score = (min(report['per_class_correct_accept_rate'].values()), report['correct_accept_rate'])
                if score > best_score:
                    best, best_report, best_score = candidate, report, score
    if best is None:
        # Keep a non-deployable, fully described artifact for diagnosing insufficient data.
        base['thresholds'] = {key: {'confidence': 1.0, 'margin': 1.0, 'distance': 0.0} for key in ZODIAC_KEYS}
        return base, evaluate(validation, base)
    return best, best_report


def passes_release(report: dict, *, min_recall: float = 0.8, max_far: float = 0.0) -> bool:
    return (all(report['counts'][key]['count'] > 0 for key in CLASS_NAMES)
            and report['false_accept_rate'] <= max_far
            and report['wrong_zodiac_rate'] == 0
            and min(report['per_class_correct_accept_rate'].values()) >= min_recall)
