"""Independent, reference-free features for candidate-solution ranking."""
from __future__ import annotations

import ast
import math
import operator
import re
from fractions import Fraction

import numpy as np

from common import parse_answer


NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SIMPLE_EXPRESSION = re.compile(r"[-+*/().\d\s]+")
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
             ast.Mult: operator.mul, ast.Div: operator.truediv}
CUES = ('therefore', 'however', 'actually', 'mistake', 'check', 'remaining',
        'total', 'each', 'per', 'more', 'less', 'percent', 'average',
        'difference', 'combined', 'left')


def parsed_answers(bank):
    return [[parse_answer(candidate['text']) for candidate in row['candidates']]
            for row in bank]


def numeric_value(answer):
    try:
        value = float(Fraction(answer))
        return value if math.isfinite(value) and abs(value) < 1e18 else 0.
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return 0.


def safe_arithmetic(expression):
    """Evaluate a small numeric expression without executing candidate code."""
    try:
        tree = ast.parse(expression, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 40:
            return None

        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
                value = OPERATORS[type(node.op)](visit(node.left), visit(node.right))
                if not math.isfinite(value) or abs(value) > 1e18:
                    raise ValueError('unbounded result')
                return value
            raise ValueError('not numeric arithmetic')

        return visit(tree)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def normalize_equations(text):
    value = text.replace(',', '').replace('×', '*').replace('÷', '/').replace('−', '-')
    value = value.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    for _ in range(4):
        updated = re.sub(r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}',
                         r'((\1)/(\2))', value)
        if updated == value:
            break
        value = updated
    value = re.sub(r'(?<=\d)\s*%', '/100', value)
    return value


def boundary_value(fragment, take_last):
    matches = list(SIMPLE_EXPRESSION.finditer(fragment))
    if not matches:
        return None
    match = matches[-1] if take_last else matches[0]
    expression = match.group().strip().strip('$').strip()
    if not expression or not re.search(r'\d', expression):
        return None
    candidates = [expression]
    if take_last:
        candidates.extend(expression[i:] for i, char in enumerate(expression)
                          if char.isdigit())
    else:
        candidates.extend(expression[:i] for i in range(len(expression), 0, -1))
    for candidate in candidates[:12]:
        result = safe_arithmetic(candidate.strip().strip('.'))
        if result is not None:
            return result
    return None


def equation_audit(text):
    checked = 0
    residuals = []
    for line in normalize_equations(text).splitlines():
        parts = line.split('=')
        for left, right in zip(parts, parts[1:]):
            a = boundary_value(left, True)
            b = boundary_value(right, False)
            if a is None or b is None:
                continue
            checked += 1
            residuals.append(abs(a - b) / max(1., abs(a), abs(b)))
    residuals = np.asarray(residuals, dtype=np.float32)
    bad = residuals > .005
    return np.asarray([
        np.log1p(checked), np.log1p(int(bad.sum())),
        float(bad.mean()) if checked else 0.,
        float(residuals.mean()) if checked else 0.,
        float(residuals.max()) if checked else 0.,
        np.log1p(int((~bad).sum())) if checked else 0.,
    ], dtype=np.float32)


def descriptor_features(bank, arrays):
    """Small interpretable text, answer-group and trace-audit features."""
    groups = parsed_answers(bank)
    embeddings = arrays['embeddings'].astype(np.float32)
    group_mean = embeddings.mean(axis=1)
    output = []
    for row_index, (row, parsed) in enumerate(zip(bank, groups)):
        question = row['question']
        qwords = set(word.lower() for word in WORD.findall(question))
        qnumbers = set(token.replace(',', '') for token in NUMBER.findall(question))
        unique_answers = set(value for value in parsed if value is not None)
        for candidate_index, (candidate, answer) in enumerate(zip(row['candidates'], parsed)):
            text = candidate['text']
            lower = text.lower()
            words = [word.lower() for word in WORD.findall(text)]
            word_set = set(words)
            numbers = [token.replace(',', '') for token in NUMBER.findall(text)]
            answer_value = numeric_value(answer)
            support = sum(answer is not None and answer == other for other in parsed)
            embedding = embeddings[row_index, candidate_index]
            mean = group_mean[row_index]
            cosine = float(np.dot(embedding, mean) /
                           max(1e-8, np.linalg.norm(embedding) * np.linalg.norm(mean)))
            overlap = len(qwords & word_set) / max(1, len(qwords))
            number_coverage = len(qnumbers & set(numbers)) / max(1, len(qnumbers))
            output.append(np.asarray([
                np.log1p(len(text)), np.log1p(len(words)), np.log1p(text.count('\n')),
                np.log1p(len(numbers)), np.log1p(len(set(numbers))),
                float(answer is not None), support / 4., len(unique_answers) / 4.,
                np.log1p(abs(answer_value)), float(answer_value < 0),
                float(answer_value == 0), float(answer_value == int(answer_value)),
                np.log1p(sum(answer is not None and answer in token for token in numbers)),
                overlap, number_coverage, cosine,
                np.log1p(np.linalg.norm(embedding - mean)), np.log1p(np.linalg.norm(embedding)),
                float('####' in text), float('\\boxed' in text),
                *[np.log1p(text.count(symbol)) for symbol in
                  ('=', '+', '-', '*', '/', '\\frac', '%', '$', '?', ':')],
                *[np.log1p(lower.count(cue)) for cue in CUES],
                *equation_audit(text),
            ], dtype=np.float32))
    return np.asarray(output, dtype=np.float32).reshape(len(bank), 4, -1)


def embedding_block(arrays, mode):
    embeddings = arrays['embeddings'].astype(np.float32)
    mean = embeddings.mean(axis=1, keepdims=True)
    if mode == 'raw':
        return embeddings
    if mode == 'product':
        return embeddings * mean
    if mode == 'squared_distance':
        return (embeddings - mean) ** 2
    if mode == 'signed_context':
        scale = np.sqrt(np.mean((embeddings - mean) ** 2, axis=1, keepdims=True) + .05)
        return (embeddings - mean) / scale
    raise ValueError(f'unknown embedding mode: {mode}')


def answer_group_choice(scores, bank, support_bonus=0.):
    scores = np.asarray(scores).reshape(len(bank), 4)
    choices = []
    for row_scores, parsed in zip(scores, parsed_answers(bank)):
        groups = {}
        for answer in dict.fromkeys(value for value in parsed if value is not None):
            members = [index for index, value in enumerate(parsed) if value == answer]
            groups[answer] = float(row_scores[members].mean() +
                                   support_bonus * np.log(len(members)))
        answer = max(groups, key=groups.get)
        choices.append(int(next(index for index, value in enumerate(parsed) if value == answer)))
    return choices
