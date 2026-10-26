"""Train compact pairwise rankers using only official training supervision."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import answer_group_choice, descriptor_features, embedding_block, parsed_answers


SEED = 7506
MODES = ('raw', 'product', 'squared_distance', 'signed_context', 'descriptors')


def labels(bank, split):
    references = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    result = []
    for row, reference in zip(bank, references):
        assert row['id'] == reference['id'] and row['question'] == reference['question']
        gold = parse_answer(reference['answer'])
        result.append([parse_answer(candidate['text']) == gold and gold is not None
                       for candidate in row['candidates']])
    return np.asarray(result, dtype=np.int8)


def pair_indices(bank, y):
    positive, negative, weights = [], [], []
    for question, (parsed, target) in enumerate(zip(parsed_answers(bank), y)):
        valid_answers = set(value for value in parsed if value is not None)
        if len(valid_answers) < 2:
            continue
        local = []
        for good in np.flatnonzero(target):
            for bad in np.flatnonzero((target == 0) & np.asarray([v is not None for v in parsed])):
                good_support = parsed.count(parsed[good])
                bad_support = parsed.count(parsed[bad])
                local.append((4 * question + good, 4 * question + bad,
                              1. / (good_support * bad_support)))
        total = sum(item[2] for item in local)
        for good, bad, weight in local:
            positive.append(good)
            negative.append(bad)
            weights.append(weight / max(total, 1e-8))
    weights = np.asarray(weights, dtype=np.float32)
    weights *= len(weights) / weights.sum()
    return np.asarray(positive), np.asarray(negative), weights


def fit_pairwise(features, positive, negative, pair_weight, regularization,
                 steps=500, learning_rate=.035):
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    scale = flat.std(axis=0).astype(np.float32)
    scale[scale < 1e-3] = 1.
    differences = ((flat[positive] - flat[negative]) / scale).astype(np.float32)
    weight = np.zeros(flat.shape[1], dtype=np.float32)
    first = np.zeros_like(weight)
    second = np.zeros_like(weight)
    for step in range(1, steps + 1):
        margin = np.clip(differences @ weight, -30, 30)
        factor = -1. / (1. + np.exp(margin))
        gradient = differences.T @ (pair_weight * factor) / len(pair_weight)
        gradient += regularization * weight
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient * gradient
        weight -= learning_rate * (first / (1 - .9 ** step)) / (
            np.sqrt(second / (1 - .999 ** step)) + 1e-8)
    return {'weight': weight, 'scale': scale,
            'regularization': regularization, 'steps': steps}


def component_scores(model, features):
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    logits = (flat / model['scale']) @ model['weight']
    return logits.reshape(features.shape[0], 4)


def accuracy(y, choices):
    return float(100 * y[np.arange(len(y)), choices].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/cleanroom'))
    parser.add_argument('--records', type=Path, default=Path('results/cleanroom'))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing checkpoint.')
    args.output.mkdir(parents=True)
    args.records.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_bank, train_arrays = load_inputs('train')
    validation_bank, validation_arrays = load_inputs('validation')
    train_y = labels(train_bank, 'train')
    validation_y = labels(validation_bank, 'validation')
    positive, negative, pair_weight = pair_indices(train_bank, train_y)
    train_descriptors = descriptor_features(train_bank, train_arrays)
    validation_descriptors = descriptor_features(validation_bank, validation_arrays)

    models = []
    validation_logits = []
    experiments = []
    regularization_grid = (.0003, .001, .003, .01)
    for mode in MODES:
        train_features = (train_descriptors if mode == 'descriptors'
                          else embedding_block(train_arrays, mode))
        validation_features = (validation_descriptors if mode == 'descriptors'
                               else embedding_block(validation_arrays, mode))
        best = None
        for regularization in regularization_grid:
            tick = time.perf_counter()
            model = fit_pairwise(train_features, positive, negative, pair_weight,
                                 regularization)
            model['output_scale'] = max(
                float(component_scores(model, train_features).std()), 1e-6)
            logits = component_scores(model, validation_features)
            choices = answer_group_choice(logits, validation_bank, 0.)
            record = {'mode': mode, 'regularization': regularization,
                      'validation_accuracy': accuracy(validation_y, choices),
                      'fit_seconds': time.perf_counter() - tick}
            experiments.append(record)
            print(json.dumps(record), flush=True)
            key = (record['validation_accuracy'], -regularization)
            if best is None or key > best[0]:
                best = (key, model, logits, record)
        _, model, logits, record = best
        models.append({'mode': mode, **model})
        validation_logits.append(logits / model['output_scale'])

    # Small, auditable convex search over at most three independently trained blocks.
    candidates = []
    for size in (1, 2, 3):
        for indices in itertools.combinations(range(len(MODES)), size):
            for counts in itertools.product((1, 2, 4), repeat=size):
                if np.gcd.reduce(counts) != 1:
                    continue
                weights = np.asarray(counts, dtype=np.float64)
                weights /= weights.sum()
                scores = sum(weight * validation_logits[index]
                             for weight, index in zip(weights, indices))
                for support_bonus in (0., .05, .1, .2, .35, .5):
                    choices = answer_group_choice(scores, validation_bank, support_bonus)
                    candidates.append((accuracy(validation_y, choices), indices,
                                       weights.tolist(), support_bonus, choices, scores))
    selected = max(candidates, key=lambda item: (item[0], -len(item[1]), -item[3]))
    score, indices, weights, support_bonus, choices, scores = selected
    selected_models = [models[index] for index in indices]
    config = {'version': 'cleanroom-v1', 'seed': SEED,
              'modes': [MODES[index] for index in indices],
              'weights': weights, 'support_bonus': support_bonus,
              'validation_accuracy': score, 'fit_split': 'train',
              'selection_split': 'validation'}
    np.savez(args.output/'model.npz',
             **{f'{key}_{i}': value for i, model in enumerate(selected_models)
                for key, value in (('weight', model['weight']), ('scale', model['scale']))})
    with np.load(args.output/'model.npz', allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays.update({f'output_scale_{i}': np.asarray(model['output_scale'], dtype=np.float32)
                   for i, model in enumerate(selected_models)})
    np.savez(args.output/'model.npz', **arrays)
    (args.output/'config.json').write_text(json.dumps(config, indent=2) + '\n')
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(args.output.iterdir())}
    (args.output/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')

    unique = np.asarray([len(set(value for value in row if value is not None))
                         for row in parsed_answers(validation_bank)])
    evidence = {'seed': SEED, 'python': platform.python_version(),
                'numpy': np.__version__, 'threads': 4,
                'training_pairs': len(positive), 'experiments': experiments,
                'ensemble_settings_evaluated': len(candidates), 'selected': config,
                'validation_correct': int(validation_y[np.arange(len(validation_y)), choices].sum()),
                'validation_disagreement_accuracy': accuracy(
                    validation_y[unique >= 2], np.asarray(choices)[unique >= 2]),
                'asset_bytes': sum(path.stat().st_size for path in args.output.iterdir()),
                'total_wall_seconds': time.perf_counter() - started,
                'test_accessed': False}
    (args.records/'experiments.json').write_text(json.dumps(evidence, indent=2) + '\n')
    np.savez_compressed(args.records/'validation_scores.npz', scores=scores,
                        components=np.stack(validation_logits), choices=choices,
                        labels=validation_y)
    print(json.dumps({'selected': config,
                      'validation_correct': evidence['validation_correct'],
                      'validation_disagreement_accuracy': evidence['validation_disagreement_accuracy'],
                      'asset_bytes': evidence['asset_bytes']}, indent=2))


if __name__ == '__main__':
    main()
