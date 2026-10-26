"""Train small CPU-only neural rankers and export NumPy inference weights.

The network is initialized randomly and trained solely on the official training
split.  PyTorch is a training dependency only; submission-time inference uses
the arrays written by this script.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from torch import nn

from common import load_inputs
from features import answer_group_choice, descriptor_features, parsed_answers
from train_method import accuracy, labels, pair_indices


SEED = 7546
SPECS = (
    {'name': 'context_tanh_32', 'width': 32, 'activation': 'tanh',
     'relative': True, 'descriptors': False},
    {'name': 'context_tanh_64', 'width': 64, 'activation': 'tanh',
     'relative': True, 'descriptors': False},
    {'name': 'context_relu_64', 'width': 64, 'activation': 'relu',
     'relative': True, 'descriptors': False},
    {'name': 'context_descriptor_tanh_64', 'width': 64, 'activation': 'tanh',
     'relative': True, 'descriptors': True},
)


class Ranker(nn.Module):
    def __init__(self, dimension, width, activation):
        super().__init__()
        self.hidden = nn.Linear(dimension, width)
        self.output = nn.Linear(width, 1, bias=False)
        self.activation = activation

    def forward(self, x):
        hidden = self.hidden(x)
        hidden = torch.tanh(hidden) if self.activation == 'tanh' else torch.relu(hidden)
        return self.output(hidden).squeeze(1)


def question_split(bank, y):
    parsed = parsed_answers(bank)
    hard = np.asarray([index for index, row in enumerate(parsed)
                       if len(set(value for value in row if value is not None)) >= 2])
    strata = [(len(set(value for value in parsed[index] if value is not None)),
               int(y[index].sum())) for index in hard]
    rng = np.random.default_rng(SEED)
    fit, dev = [], []
    for stratum in sorted(set(strata)):
        indices = hard[np.asarray([value == stratum for value in strata])]
        rng.shuffle(indices)
        boundary = max(1, int(round(.2 * len(indices))))
        dev.extend(indices[:boundary]); fit.extend(indices[boundary:])
    return np.asarray(sorted(fit)), np.asarray(sorted(dev))


def input_statistics(train_bank, train_arrays):
    embeddings = train_arrays['embeddings'].astype(np.float32)
    embedding_mean = embeddings.mean(axis=(0, 1), keepdims=True)
    embedding_scale = embeddings.std(axis=(0, 1), keepdims=True)
    embedding_scale[embedding_scale < .02] = 1.
    descriptors = descriptor_features(train_bank, train_arrays).astype(np.float32)
    descriptor_mean = descriptors.mean(axis=(0, 1), keepdims=True)
    descriptor_scale = descriptors.std(axis=(0, 1), keepdims=True)
    descriptor_scale[descriptor_scale < 1e-3] = 1.
    return embedding_mean, embedding_scale, descriptor_mean, descriptor_scale


def inputs(bank, arrays, statistics, spec):
    embedding_mean, embedding_scale, descriptor_mean, descriptor_scale = statistics
    embeddings = np.clip(
        (arrays['embeddings'].astype(np.float32) - embedding_mean) / embedding_scale,
        -8., 8.)
    blocks = [embeddings]
    if spec['relative']:
        blocks.append(embeddings - embeddings.mean(axis=1, keepdims=True))
    if spec['descriptors']:
        descriptors = descriptor_features(bank, arrays).astype(np.float32)
        blocks.append(np.clip((descriptors - descriptor_mean) / descriptor_scale, -8., 8.))
    return np.concatenate(blocks, axis=2).reshape(-1, sum(block.shape[2] for block in blocks))


@torch.no_grad()
def predict(model, x, questions=None):
    if questions is None:
        rows = np.arange(len(x))
    else:
        rows = (questions[:, None] * 4 + np.arange(4)).reshape(-1)
    result = []
    for start in range(0, len(rows), 4096):
        batch = torch.from_numpy(x[rows[start:start + 4096]])
        result.append(model(batch).numpy())
    return np.concatenate(result).reshape(-1, 4)


def fit_model(x, positive, negative, pair_weight, dimension, spec, epochs, seed):
    torch.manual_seed(seed)
    model = Ranker(dimension, spec['width'], spec['activation'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    rng = np.random.default_rng(seed)
    x_tensor = torch.from_numpy(x)
    for _ in range(epochs):
        order = rng.permutation(len(positive))
        for start in range(0, len(order), 2048):
            take = order[start:start + 2048]
            good = model(x_tensor[positive[take]])
            bad = model(x_tensor[negative[take]])
            weights = torch.from_numpy(pair_weight[take])
            loss = (torch.nn.functional.softplus(-(good - bad)) * weights).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
    return model


def select_epochs(x, positive, negative, pair_weight, train_questions, dev_questions,
                  train_bank, train_y, spec):
    pair_question = positive // 4
    keep = np.isin(pair_question, train_questions)
    model = Ranker(x.shape[1], spec['width'], spec['activation'])
    torch.manual_seed(SEED)
    model = Ranker(x.shape[1], spec['width'], spec['activation'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    rng = np.random.default_rng(SEED)
    x_tensor = torch.from_numpy(x)
    p, n, w = positive[keep], negative[keep], pair_weight[keep]
    best = (-1., 10, None); stale = 0
    for epoch in range(1, 201):
        order = rng.permutation(len(p))
        for start in range(0, len(order), 2048):
            take = order[start:start + 2048]
            good = model(x_tensor[p[take]]); bad = model(x_tensor[n[take]])
            weights = torch.from_numpy(w[take])
            loss = (torch.nn.functional.softplus(-(good - bad)) * weights).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        if epoch % 5:
            continue
        dev_scores = predict(model, x, dev_questions)
        dev_bank = [train_bank[index] for index in dev_questions]
        choices = answer_group_choice(dev_scores, dev_bank, .1)
        value = accuracy(train_y[dev_questions], choices)
        if value > best[0]:
            best = (value, epoch, {key: tensor.detach().clone()
                                   for key, tensor in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= 10:
                break
    return best[1], best[0]


def export_model(path, model, statistics, spec, output_scale):
    embedding_mean, embedding_scale, descriptor_mean, descriptor_scale = statistics
    np.savez(path,
             hidden_weight=model.hidden.weight.detach().numpy(),
             hidden_bias=model.hidden.bias.detach().numpy(),
             output_weight=model.output.weight.detach().numpy().reshape(-1),
             embedding_mean=embedding_mean,
             embedding_scale=embedding_scale,
             descriptor_mean=descriptor_mean,
             descriptor_scale=descriptor_scale,
             output_scale=np.asarray(output_scale, dtype=np.float32),
             relative=np.asarray(int(spec['relative']), dtype=np.int8),
             descriptors=np.asarray(int(spec['descriptors']), dtype=np.int8),
             activation=np.asarray(spec['activation']))


def main():
    output = Path('results/mlp'); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    train_bank, train_arrays = load_inputs('train')
    validation_bank, validation_arrays = load_inputs('validation')
    train_y = labels(train_bank, 'train'); validation_y = labels(validation_bank, 'validation')
    positive, negative, pair_weight = pair_indices(train_bank, train_y)
    fit_questions, dev_questions = question_split(train_bank, train_y)
    statistics = input_statistics(train_bank, train_arrays)
    records, validation_scores, models = [], [], []
    for offset, spec in enumerate(SPECS):
        tick = time.perf_counter()
        train_x = inputs(train_bank, train_arrays, statistics, spec)
        validation_x = inputs(validation_bank, validation_arrays, statistics, spec)
        epochs, internal_accuracy = select_epochs(
            train_x, positive, negative, pair_weight, fit_questions, dev_questions,
            train_bank, train_y, spec)
        model = fit_model(train_x, positive, negative, pair_weight, train_x.shape[1],
                          spec, epochs, SEED + offset)
        train_score = predict(model, train_x)
        output_scale = max(float(train_score.std()), 1e-6)
        score = predict(model, validation_x) / output_scale
        choices = answer_group_choice(score, validation_bank, .1)
        record = {**spec, 'epochs': epochs, 'internal_dev_accuracy': internal_accuracy,
                  'validation_accuracy': accuracy(validation_y, choices),
                  'fit_seconds': time.perf_counter() - tick}
        records.append(record); validation_scores.append(score)
        models.append((model, spec, output_scale))
        print(json.dumps(record), flush=True)
    best = max(range(len(records)), key=lambda index: records[index]['validation_accuracy'])
    export_model(output/'model.npz', models[best][0], statistics,
                 models[best][1], models[best][2])
    np.savez_compressed(output/'validation_scores.npz',
                        scores=np.stack(validation_scores), labels=validation_y)
    evidence = {'seed': SEED, 'fit_split': 'train', 'selection_split': 'validation',
                'training_device': 'cpu', 'internal_fit_questions': len(fit_questions),
                'internal_dev_questions': len(dev_questions), 'experiments': records,
                'selected': records[best], 'asset_bytes': (output/'model.npz').stat().st_size,
                'total_wall_seconds': time.perf_counter() - started, 'test_accessed': False,
                'gpu_used': False}
    (output/'experiments.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
