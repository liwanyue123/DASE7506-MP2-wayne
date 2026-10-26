"""Deterministic hashed lexical evidence with NumPy-only inference."""
from __future__ import annotations

from collections import Counter
import re

import numpy as np


WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")


def stable_hash(token):
    value = 2166136261
    for byte in token.encode('utf-8'):
        value ^= byte
        value = (value * 16777619) & 0xffffffff
    return value


def tokens(question, text):
    question_words = set(word.lower() for word in WORDS.findall(question))
    words = [word.lower() for word in WORDS.findall(text)]
    result = ['w=' + word for word in words]
    result.extend('b=' + left + '_' + right for left, right in zip(words, words[1:]))
    result.extend('qoverlap=' + word for word in words if word in question_words)
    result.extend('prefix=' + word for word in words[:24])
    return result


def sparse_row(question, text, dimension):
    counts = Counter(stable_hash(token) % dimension for token in tokens(question, text))
    indices = np.asarray(sorted(counts), dtype=np.int32)
    values = np.asarray([1. + np.log(counts[int(index)]) for index in indices],
                        dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm:
        values /= norm
    return indices, values


def bank_rows(bank, dimension):
    return [sparse_row(row['question'], candidate['text'], dimension)
            for row in bank for candidate in row['candidates']]


def lexical_scores(coefficient, bank):
    rows = bank_rows(bank, len(coefficient))
    return np.asarray([float(coefficient[index] @ value) if len(index) else 0.
                       for index, value in rows], dtype=np.float32).reshape(len(bank), 4)
