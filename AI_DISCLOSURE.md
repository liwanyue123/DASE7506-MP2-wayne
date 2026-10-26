# AI assistance disclosure

OpenAI Codex provided substantive assistance with independent method design,
feature engineering, implementation, debugging, training and validation
experiments, resource measurement, analysis and documentation. The work began
from the official course repository and did not copy another student's code,
weights, predictions or report. The submitting student remains responsible for
reviewing, understanding and accurately presenting the final implementation.

No external dataset, additional pretrained weight, evaluation-time LLM/API
call, answer lookup or question-ID rule is used. All learned parameters are fit
from the supplied training split; validation is used for development and model
selection.

Codex was also used to operate the student-provided compute server. Although an
H800 was present, all training and evaluation commands explicitly disabled CUDA
and used four CPU threads. The public test reference was first accessed only
after the final source and asset hashes were frozen. Its result is reported but
was not used for any subsequent model or hyperparameter change.
