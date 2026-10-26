"""NumPy-only runtime for the frozen independent candidate selector."""
import json
from pathlib import Path

import numpy as np

from features import answer_group_choice, descriptor_features, embedding_block


def linear_scores(bank, arrays, payload, config, descriptors):
    scores = np.zeros((len(bank), 4), dtype=np.float32)
    for index, (mode, mixture_weight) in enumerate(zip(
            config['linear_modes'], config['linear_component_weights'])):
        features = descriptors if mode == 'descriptors' else embedding_block(arrays, mode)
        flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
        logits = ((flat / payload[f'linear_scale_{index}']) @
                  payload[f'linear_weight_{index}'])
        logits = logits.reshape(len(bank), 4)
        logits /= float(payload[f'linear_output_scale_{index}'])
        scores += np.float32(mixture_weight) * logits
    return scores


def mlp_scores(arrays, descriptors, payload):
    embeddings = arrays['embeddings'].astype(np.float32)
    embeddings = np.clip(
        (embeddings - payload['embedding_mean']) / payload['embedding_scale'], -8., 8.)
    relative = embeddings - embeddings.mean(axis=1, keepdims=True)
    descriptors = np.clip(
        (descriptors - payload['descriptor_mean']) / payload['descriptor_scale'], -8., 8.)
    features = np.concatenate([embeddings, relative, descriptors], axis=2)
    flat = features.reshape(-1, features.shape[-1])
    hidden_weight = payload['mlp_hidden_weight'].astype(np.float32)
    hidden_bias = payload['mlp_hidden_bias'].astype(np.float32)
    output_weight = payload['mlp_output_weight'].astype(np.float32)
    scores = np.zeros(len(flat), dtype=np.float32)
    for member in range(len(hidden_weight)):
        hidden = np.tanh(flat @ hidden_weight[member].T + hidden_bias[member])
        scores += (hidden @ output_weight[member]) / float(payload['mlp_output_scale'][member])
    return (scores / len(hidden_weight)).reshape(len(embeddings), 4)


def candidate_scores(bank, arrays, checkpoint):
    if not bank:
        return np.empty((0, 4), dtype=np.float64), 0.
    root = Path(checkpoint)
    config = json.loads((root/'config.json').read_text())
    descriptors = descriptor_features(bank, arrays)
    with np.load(root/'model.npz', allow_pickle=False) as payload:
        linear = linear_scores(bank, arrays, payload, config, descriptors)
        neural = mlp_scores(arrays, descriptors, payload)
    scores = config['linear_weight'] * linear + config['mlp_weight'] * neural
    return scores, config['answer_support_bonus']


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    scores, support_bonus = candidate_scores(bank, arrays, checkpoint)
    return answer_group_choice(scores, bank, support_bonus)
