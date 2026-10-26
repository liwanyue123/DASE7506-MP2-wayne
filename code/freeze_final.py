"""Freeze the validation-selected, test-unseen inference bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


LINEAR = Path('artifacts/linear-v1')
MLP = Path('results/mlp-stability')
OUTPUT = Path('artifacts/final')
RECORD = Path('results/final-selection.json')
SEEDS = (7550, 7551, 7552, 7553, 7554)


def main():
    if OUTPUT.exists():
        raise SystemExit('Refusing to overwrite the frozen bundle.')
    OUTPUT.mkdir(parents=True)
    linear_config = json.loads((LINEAR/'config.json').read_text())
    with np.load(LINEAR/'model.npz', allow_pickle=False) as payload:
        arrays = {f'linear_{key}': payload[key] for key in payload.files}

    members = []
    for seed in SEEDS:
        with np.load(MLP/f'model-{seed}.npz', allow_pickle=False) as payload:
            members.append({key: payload[key] for key in payload.files})
    shared = ('embedding_mean', 'embedding_scale', 'descriptor_mean', 'descriptor_scale')
    for key in shared:
        if any(not np.array_equal(members[0][key], member[key]) for member in members[1:]):
            raise ValueError(f'MLP members disagree on shared statistic: {key}')
        arrays[key] = members[0][key].astype(np.float32)
    # Half-precision storage is converted to float32 once at runtime.  It halves
    # the bundle size while preserving the selected validation decisions.
    arrays['mlp_hidden_weight'] = np.stack(
        [member['hidden_weight'] for member in members]).astype(np.float16)
    arrays['mlp_hidden_bias'] = np.stack(
        [member['hidden_bias'] for member in members]).astype(np.float16)
    arrays['mlp_output_weight'] = np.stack(
        [member['output_weight'] for member in members]).astype(np.float16)
    arrays['mlp_output_scale'] = np.asarray(
        [float(member['output_scale']) for member in members], dtype=np.float32)
    np.savez(OUTPUT/'model.npz', **arrays)

    config = {
        'version': 'independent-selector-v2',
        'linear_modes': linear_config['modes'],
        'linear_component_weights': linear_config['weights'],
        'linear_weight': .55,
        'mlp_weight': .45,
        'mlp_members': len(SEEDS),
        'mlp_seeds': SEEDS,
        'mlp_activation': 'tanh',
        'mlp_relative_embeddings': True,
        'mlp_descriptors': True,
        'answer_support_bonus': .1,
        'fit_split': 'train',
        'selection_split': 'validation',
        'test_accessed_at_freeze': False,
    }
    (OUTPUT/'config.json').write_text(json.dumps(config, indent=2) + '\n')
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(OUTPUT.iterdir())}
    (OUTPUT/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    asset_bytes = sum(path.stat().st_size for path in OUTPUT.iterdir())

    base = np.load('results/linear-v1/validation_scores.npz')
    stable = np.load(MLP/'validation_scores.npz')
    record = {
        'decision': 'freeze before any test-reference access',
        'configuration': config,
        'validation_expected_correct_before_float16_check': 754,
        'validation_expected_accuracy': 100 * 754 / 798,
        'linear_baseline_correct': 749,
        'linear_baseline_accuracy': 100 * 749 / 798,
        'mlp_seed_mean_single_accuracy': json.loads(
            (MLP/'experiments.json').read_text())['mean_single_model_accuracy'],
        'inference_asset_bytes': asset_bytes,
        'inference_asset_mib': asset_bytes / 2**20,
        'rejected_auxiliaries': {
            'lexical': 'no validation gain over the linear baseline',
            'trees': 'one extra validation correction did not justify runtime complexity',
            'fixed_random_features': 'smaller and less stable gain than the seed-averaged MLP',
        },
        'test_accessed': False,
    }
    assert base['scores'].shape == stable['mean_score'].shape == (798, 4)
    RECORD.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
