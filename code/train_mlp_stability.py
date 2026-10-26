"""Repeat the selected MLP specification across seeds to measure stability."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

from common import load_inputs
from features import answer_group_choice
from train_method import accuracy, labels, pair_indices
from train_mlp import (export_model, fit_model, input_statistics, inputs, predict)


SEEDS = (7550, 7551, 7552, 7553, 7554)
SPEC = {'name': 'context_descriptor_tanh_64', 'width': 64,
        'activation': 'tanh', 'relative': True, 'descriptors': True}
EPOCHS = 100


def main():
    output = Path('results/mlp-stability'); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    train_bank, train_arrays = load_inputs('train')
    validation_bank, validation_arrays = load_inputs('validation')
    train_y = labels(train_bank, 'train'); validation_y = labels(validation_bank, 'validation')
    positive, negative, pair_weight = pair_indices(train_bank, train_y)
    statistics = input_statistics(train_bank, train_arrays)
    train_x = inputs(train_bank, train_arrays, statistics, SPEC)
    validation_x = inputs(validation_bank, validation_arrays, statistics, SPEC)
    records, scores = [], []
    for seed in SEEDS:
        tick = time.perf_counter()
        model = fit_model(train_x, positive, negative, pair_weight,
                          train_x.shape[1], SPEC, EPOCHS, seed)
        output_scale = max(float(predict(model, train_x).std()), 1e-6)
        validation_score = predict(model, validation_x) / output_scale
        choices = answer_group_choice(validation_score, validation_bank, .1)
        record = {'seed': seed, 'validation_accuracy': accuracy(validation_y, choices),
                  'fit_seconds': time.perf_counter() - tick}
        records.append(record); scores.append(validation_score)
        export_model(output/f'model-{seed}.npz', model, statistics, SPEC, output_scale)
        print(json.dumps(record), flush=True)
    mean_score = np.mean(scores, axis=0)
    mean_choices = answer_group_choice(mean_score, validation_bank, .1)
    evidence = {'seeds': SEEDS, 'specification': SPEC, 'epochs': EPOCHS,
                'experiments': records,
                'mean_single_model_accuracy': float(np.mean(
                    [record['validation_accuracy'] for record in records])),
                'standard_deviation': float(np.std(
                    [record['validation_accuracy'] for record in records])),
                'seed_ensemble_accuracy': accuracy(validation_y, mean_choices),
                'asset_bytes': sum(path.stat().st_size for path in output.glob('model-*.npz')),
                'total_wall_seconds': time.perf_counter() - started,
                'training_device': 'cpu', 'gpu_used': False, 'test_accessed': False}
    np.savez_compressed(output/'validation_scores.npz', scores=np.stack(scores),
                        mean_score=mean_score, labels=validation_y)
    (output/'experiments.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
