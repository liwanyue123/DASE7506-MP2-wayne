"""Train CPU-only nonlinear semantic/trace rankers on official train data."""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from common import load_inputs
from features import answer_group_choice, parsed_answers
from train_method import accuracy, labels
from tree_features import fit_projection, tree_features


SEED = 7536


def hard_questions(bank):
    return np.asarray([len(set(value for value in row if value is not None)) >= 2
                       for row in parsed_answers(bank)])


def question_split(bank, y):
    hard = np.flatnonzero(hard_questions(bank))
    strata = [(len(set(value for value in parsed_answers([bank[index]])[0]
                       if value is not None)), int(y[index].sum())) for index in hard]
    rng = np.random.default_rng(SEED)
    fit, dev = [], []
    for stratum in sorted(set(strata)):
        indices = hard[np.asarray([value == stratum for value in strata])]
        rng.shuffle(indices)
        boundary = max(1, int(round(.2 * len(indices))))
        dev.extend(indices[:boundary]); fit.extend(indices[boundary:])
    return np.asarray(sorted(fit)), np.asarray(sorted(dev)), hard


def rows(questions):
    return (questions[:, None] * 4 + np.arange(4)).reshape(-1)


def probability(logits):
    return 1. / (1. + np.exp(-np.clip(logits, -30, 30)))


def main():
    output = Path('results/trees'); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_bank, train_arrays = load_inputs('train')
    validation_bank, validation_arrays = load_inputs('validation')
    train_y = labels(train_bank, 'train'); validation_y = labels(validation_bank, 'validation')
    projection = fit_projection(train_arrays)
    train_x = tree_features(train_bank, train_arrays, projection)
    validation_x = tree_features(validation_bank, validation_arrays, projection)
    fit, dev, hard = question_split(train_bank, train_y)
    fit_rows, dev_rows, hard_rows = rows(fit), rows(dev), rows(hard)

    specifications = [('PairLogit', 4, 20.), ('PairLogit', 6, 40.),
                      ('PairLogit', 8, 80.), ('Logloss', 4, 20.),
                      ('Logloss', 6, 40.)]
    records = []; predictions = []
    for objective, depth, l2 in specifications:
        tick = time.perf_counter()
        parameters = dict(iterations=1400, depth=depth, learning_rate=.035,
                          l2_leaf_reg=l2, random_seed=SEED, thread_count=4,
                          verbose=False, allow_writing_files=False,
                          loss_function=objective)
        if objective == 'PairLogit':
            pilot = CatBoostRanker(**parameters).fit(
                Pool(train_x[fit_rows], train_y[fit].reshape(-1),
                     group_id=np.repeat(np.arange(len(fit)), 4)),
                eval_set=Pool(train_x[dev_rows], train_y[dev].reshape(-1),
                              group_id=np.repeat(np.arange(len(dev)), 4)),
                early_stopping_rounds=100)
            iterations = max(100, pilot.best_iteration_ + 1)
            parameters['iterations'] = iterations
            model = CatBoostRanker(**parameters).fit(
                Pool(train_x[hard_rows], train_y[hard].reshape(-1),
                     group_id=np.repeat(np.arange(len(hard)), 4)))
            train_score = probability(model.predict(train_x[hard_rows]))
            validation_score = probability(model.predict(validation_x)).reshape(-1, 4)
        else:
            pilot = CatBoostClassifier(**parameters).fit(
                train_x[fit_rows], train_y[fit].reshape(-1),
                eval_set=(train_x[dev_rows], train_y[dev].reshape(-1)),
                early_stopping_rounds=100)
            iterations = max(100, pilot.best_iteration_ + 1)
            parameters['iterations'] = iterations
            model = CatBoostClassifier(**parameters).fit(
                train_x[hard_rows], train_y[hard].reshape(-1))
            train_score = model.predict_proba(train_x[hard_rows])[:, 1]
            validation_score = model.predict_proba(validation_x)[:, 1].reshape(-1, 4)
        output_scale = max(float(train_score.std()), 1e-6)
        validation_score /= output_scale
        choices = answer_group_choice(validation_score, validation_bank, 0.)
        name = f'{objective.lower()}_d{depth}'
        model.save_model(str(output/f'{name}.cbm'))
        with (output/f'{name}-projection.pkl').open('wb') as handle:
            pickle.dump({'projection': projection, 'output_scale': output_scale,
                         'objective': objective}, handle, protocol=5)
        record = {'name': name, 'objective': objective, 'depth': depth,
                  'l2': l2, 'iterations': iterations,
                  'validation_accuracy': accuracy(validation_y, choices),
                  'fit_seconds': time.perf_counter() - tick}
        records.append(record); predictions.append(validation_score)
        print(json.dumps(record), flush=True)
    np.savez_compressed(output/'validation_scores.npz', scores=np.stack(predictions),
                        labels=validation_y)
    evidence = {'seed': SEED, 'fit_split': 'train', 'selection_split': 'validation',
                'feature_dimensions': train_x.shape[1], 'hard_train_questions': len(hard),
                'internal_fit_questions': len(fit), 'internal_dev_questions': len(dev),
                'experiments': records, 'total_wall_seconds': time.perf_counter() - started,
                'test_accessed': False, 'gpu_used': False}
    (output/'experiments.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
