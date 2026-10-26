"""Train a compact pairwise hashed lexical ranker on the official train split."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import load_inputs
from features import answer_group_choice
from lexical import bank_rows, lexical_scores
from train_method import accuracy, labels, pair_indices


DIMENSION = 65536
SEED = 7516


def difference_row(positive, negative):
    positive_index, positive_value = positive
    negative_index, negative_value = negative
    index = np.concatenate([positive_index, negative_index])
    value = np.concatenate([positive_value, -negative_value])
    unique, inverse = np.unique(index, return_inverse=True)
    merged = np.zeros(len(unique), dtype=np.float32)
    np.add.at(merged, inverse, value)
    keep = merged != 0
    return unique[keep], merged[keep]


def fit(rows, positive, negative, pair_weight, epochs):
    rng = np.random.default_rng(SEED)
    coefficient = np.zeros(DIMENSION, dtype=np.float32)
    accumulator = np.full(DIMENSION, 1e-4, dtype=np.float32)
    order = np.arange(len(positive))
    for epoch in range(epochs):
        rng.shuffle(order)
        for pair in order:
            index, value = difference_row(rows[positive[pair]], rows[negative[pair]])
            margin = float(coefficient[index] @ value)
            strength = pair_weight[pair] / (1. + np.exp(np.clip(margin, -30, 30)))
            gradient = -strength * value
            accumulator[index] += gradient * gradient
            coefficient[index] -= .08 * gradient / np.sqrt(accumulator[index])
        coefficient *= .998
    return coefficient


def main():
    output = Path('results/lexical'); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_bank, _ = load_inputs('train')
    validation_bank, _ = load_inputs('validation')
    train_y = labels(train_bank, 'train'); validation_y = labels(validation_bank, 'validation')
    positive, negative, pair_weight = pair_indices(train_bank, train_y)
    rows = bank_rows(train_bank, DIMENSION)
    records = []; scores = []; coefficients = []
    for epochs in (2, 5, 10, 20):
        tick = time.perf_counter()
        coefficient = fit(rows, positive, negative, pair_weight, epochs)
        validation_score = lexical_scores(coefficient, validation_bank)
        choices = answer_group_choice(validation_score, validation_bank, 0.)
        record = {'epochs': epochs, 'validation_accuracy': accuracy(validation_y, choices),
                  'fit_seconds': time.perf_counter() - tick}
        records.append(record); scores.append(validation_score); coefficients.append(coefficient)
        print(json.dumps(record), flush=True)
    best = max(range(len(records)), key=lambda index: records[index]['validation_accuracy'])
    np.savez(output/'model.npz', coefficient=coefficients[best], dimension=DIMENSION,
             output_scale=max(float(scores[best].std()), 1e-6))
    np.savez_compressed(output/'validation_scores.npz', scores=np.stack(scores),
                        labels=validation_y)
    evidence = {'seed': SEED, 'fit_split': 'train', 'selection_split': 'validation',
                'dimension': DIMENSION, 'training_pairs': len(positive),
                'experiments': records, 'selected_epochs': records[best]['epochs'],
                'asset_bytes': (output/'model.npz').stat().st_size,
                'total_wall_seconds': time.perf_counter() - started,
                'test_accessed': False}
    (output/'experiments.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
