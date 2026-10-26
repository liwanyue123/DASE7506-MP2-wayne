"""Train nonlinear random-projection rankers without pretrained weights."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import load_inputs
from features import answer_group_choice
from train_method import accuracy, component_scores, fit_pairwise, labels, pair_indices


SEED = 7526
DIMENSION = 256


def projected_inputs(arrays, mean, scale, projection):
    embeddings = arrays['embeddings'].astype(np.float32)
    standardized = np.clip((embeddings - mean) / scale, -8, 8)
    raw = standardized.reshape(-1, 1024) @ projection
    centered = standardized - standardized.mean(axis=1, keepdims=True)
    relative = centered.reshape(-1, 1024) @ projection
    return raw.reshape(len(embeddings), 4, -1), relative.reshape(len(embeddings), 4, -1)


def nonlinear_features(raw, relative, kind, phase):
    if kind == 'relu':
        return np.concatenate([np.maximum(raw, 0), np.maximum(relative, 0)], axis=2)
    if kind == 'tanh':
        return np.concatenate([np.tanh(raw), np.tanh(relative)], axis=2)
    if kind == 'cosine':
        return np.concatenate([np.cos(raw + phase), np.cos(relative + phase)], axis=2)
    if kind == 'interaction':
        return np.concatenate([raw, relative, raw * relative], axis=2)
    raise ValueError(kind)


def main():
    output = Path('results/random-evidence'); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_bank, train_arrays = load_inputs('train')
    validation_bank, validation_arrays = load_inputs('validation')
    train_y = labels(train_bank, 'train'); validation_y = labels(validation_bank, 'validation')
    positive, negative, pair_weight = pair_indices(train_bank, train_y)
    embeddings = train_arrays['embeddings'].astype(np.float32)
    mean = embeddings.mean(axis=(0, 1), keepdims=True)
    scale = embeddings.std(axis=(0, 1), keepdims=True)
    scale[scale < .02] = 1.
    rng = np.random.default_rng(SEED)
    projection = (rng.standard_normal((1024, DIMENSION)) / np.sqrt(1024)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, DIMENSION).astype(np.float32)
    train_raw, train_relative = projected_inputs(train_arrays, mean, scale, projection)
    validation_raw, validation_relative = projected_inputs(
        validation_arrays, mean, scale, projection)

    records = []; predictions = []; models = []
    for kind in ('relu', 'tanh', 'cosine', 'interaction'):
        train_features = nonlinear_features(train_raw, train_relative, kind, phase)
        validation_features = nonlinear_features(validation_raw, validation_relative, kind, phase)
        for regularization in (.001, .003, .01, .03):
            tick = time.perf_counter()
            model = fit_pairwise(train_features, positive, negative, pair_weight,
                                 regularization, steps=600, learning_rate=.025)
            model['output_scale'] = max(float(component_scores(model, train_features).std()), 1e-6)
            score = component_scores(model, validation_features) / model['output_scale']
            choices = answer_group_choice(score, validation_bank, 0.)
            record = {'kind': kind, 'regularization': regularization,
                      'validation_accuracy': accuracy(validation_y, choices),
                      'fit_seconds': time.perf_counter() - tick}
            records.append(record); predictions.append(score); models.append(model)
            print(json.dumps(record), flush=True)
    best = max(range(len(records)), key=lambda index: records[index]['validation_accuracy'])
    np.savez(output/'model.npz', projection=projection, phase=phase,
             input_mean=mean, input_scale=scale, coefficient=models[best]['weight'],
             feature_scale=models[best]['scale'], output_scale=models[best]['output_scale'])
    np.savez_compressed(output/'validation_scores.npz', scores=np.stack(predictions),
                        labels=validation_y)
    evidence = {'seed': SEED, 'fit_split': 'train', 'selection_split': 'validation',
                'projection_dimensions': DIMENSION, 'experiments': records,
                'selected': records[best], 'asset_bytes': (output/'model.npz').stat().st_size,
                'total_wall_seconds': time.perf_counter() - started, 'test_accessed': False}
    (output/'experiments.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
