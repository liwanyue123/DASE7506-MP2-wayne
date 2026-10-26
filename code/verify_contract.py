"""Reference-free behavioral checks for the frozen submission interface."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer
from submission import predict


CHECKPOINT = Path('artifacts/final')


def subset(bank, arrays, indices):
    return ([bank[index] for index in indices],
            {key: value[indices] for key, value in arrays.items()})


def main():
    bank, arrays = load_inputs('validation')
    expected = predict(bank, arrays, CHECKPOINT)

    chunks = []
    for indices in np.array_split(np.arange(len(bank)), 7):
        part_bank, part_arrays = subset(bank, arrays, indices)
        chunks.extend(predict(part_bank, part_arrays, CHECKPOINT))
    assert chunks == expected, 'predictions depend on batch boundaries'

    rng = np.random.default_rng(7560)
    order = rng.permutation(len(bank))
    ordered_bank, ordered_arrays = subset(bank, arrays, order)
    ordered = predict(ordered_bank, ordered_arrays, CHECKPOINT)
    restored = np.empty(len(bank), dtype=np.int64); restored[order] = ordered
    assert restored.tolist() == expected, 'predictions depend on question order'

    renamed = copy.deepcopy(bank[:31])
    for index, row in enumerate(renamed):
        row['id'] = f'unseen-audit-id-{index}'
    assert predict(renamed, {key: value[:31] for key, value in arrays.items()}, CHECKPOINT) == expected[:31]

    count = 73
    permuted_bank = copy.deepcopy(bank[:count])
    permuted_arrays = {key: value[:count].copy() for key, value in arrays.items()}
    permutations = []
    for question, row in enumerate(permuted_bank):
        permutation = rng.permutation(4)
        permutations.append(permutation)
        row['candidates'] = [row['candidates'][index] for index in permutation]
        for index, candidate in enumerate(row['candidates']):
            candidate['index'] = index
        for key in permuted_arrays:
            permuted_arrays[key][question] = permuted_arrays[key][question, permutation]
    permuted = predict(permuted_bank, permuted_arrays, CHECKPOINT)
    for question, choice in enumerate(permuted):
        original_choice = int(permutations[question][choice])
        before = parse_answer(bank[question]['candidates'][expected[question]]['text'])
        after = parse_answer(bank[question]['candidates'][original_choice]['text'])
        assert before == after, 'selected answer changes under candidate permutation'

    assert predict([], {'embeddings': np.empty((0, 4, 1024), dtype=np.float32)}, CHECKPOINT) == []
    asset_bytes = sum(path.stat().st_size for path in CHECKPOINT.iterdir())
    assert asset_bytes <= 16 * 2**20
    result = {'batch_independent': True, 'question_order_independent': True,
              'candidate_permutation_equivariant': True, 'ids_ignored': True,
              'empty_bank_supported': True, 'asset_bytes': asset_bytes,
              'asset_limit_bytes': 16 * 2**20, 'test_references_accessed': False}
    Path('results/contract-checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
