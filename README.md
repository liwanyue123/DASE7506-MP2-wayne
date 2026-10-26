# Independent MP2 candidate selector

This is a clean-room solution to DASE7506 MP2. It starts from the official
course repository at commit `bb801f4` and does not reuse another student's
code, checkpoint, predictions, experiments or report. See
[`CLEANROOM.md`](CLEANROOM.md) and [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).

## Result

| Split | Correct | Accuracy | Role |
|---|---:|---:|---|
| Validation | 754 / 798 | 94.4862% | development and selection |
| Test | 395 / 467 | 84.5824% | one post-freeze evaluation |

The full test prediction and official score are in
`code/results/final-test-predictions.json` and
`code/results/final-test-score.json`. The test result did not trigger any
further model selection or tuning.

## Method

The selector combines two independently trained signals:

1. A pairwise logistic ranker over the supplied 1,024-dimensional candidate
   embeddings and 52 auditable text/answer features.
2. The mean of five small one-hidden-layer pairwise rankers. They use the
   candidate embedding, its within-question centered version and the same
   52 descriptors. Each member is trained from a different random seed.

Candidate scores receive 55% linear and 45% neural weight. Candidates with the
same parsed final answer are pooled before selection, with a small answer-group
support bonus. The runtime is NumPy-only and never opens a reference answer.

## Reproduce the frozen result

Run from `code/` using Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 check_data.py
python3 verify_contract.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python3 predict.py --method submission.py --checkpoint artifacts/final \
  --split test --output results/reproduced-test-predictions.json

python3 score_predictions.py --split test \
  --predictions results/reproduced-test-predictions.json \
  --output results/reproduced-test-score.json
```

Expected test score: 395/467, or 84.58244111349036%. The frozen bundle occupies
1,366,950 bytes (1.304 MiB), including its configuration and hashes.

## Reproduce training and ablations

Use a fresh checkout so the scripts' overwrite guards remain meaningful:

```bash
python3 -m pip install -r requirements-training.txt
CUDA_VISIBLE_DEVICES='' python3 train_method.py \
  --output artifacts/linear-v1 --records results/linear-v1
CUDA_VISIBLE_DEVICES='' python3 train_lexical.py
CUDA_VISIBLE_DEVICES='' python3 train_random_evidence.py
CUDA_VISIBLE_DEVICES='' python3 train_trees.py
CUDA_VISIBLE_DEVICES='' python3 train_mlp.py
CUDA_VISIBLE_DEVICES='' python3 train_mlp_stability.py
python3 freeze_final.py
```

The final bundle is already included, so retraining is unnecessary for score
reproduction. `REPORT.md` explains the experiment design, ablations, failure
analysis and resource trade-offs. Raw experiment records are under
`code/results/`.
