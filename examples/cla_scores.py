"""Shared classification scoring for the Elo arena (pure stdlib).

Contract (2026-10-31 classification round):
- `score_cla(proba, y_true, classes)` -> dict(logloss, accuracy, bal_acc,
  auc, macro_f1). `proba`: one row per test case, values aligned to `classes`
  (the meta.json canonical str-sorted order), or a bare positive-class
  probability per row for binary (expanded internally). `y_true`: raw class
  values as they appear in the split CSV.
- positive class for binary = classes[1] in canonical order (for '0'/'1'
  labels this is '1'); ROC AUC is computed on that score and is None for
  multiclass.
- logloss is the primary board metric: a strictly proper score, unified
  across binary and multiclass. Accuracy is deliberately NOT a board metric
  (the imbalance lesson); it is reported as a diagnostic alongside balanced
  accuracy, AUC (binary) and macro-F1.
"""

import math


def expand_proba(proba):
    rows = []
    for p in proba:
        if isinstance(p, (int, float)):  # binary positive-class column
            f = float(p)
            rows.append([1.0 - f, f])
        else:
            rows.append([float(v) for v in p])
    return rows


def log_loss(proba, y_idx):
    eps = 1e-15
    total = 0.0
    for row, yi in zip(proba, y_idx):
        p = min(max(row[yi], eps), 1.0 - eps)
        total -= math.log(p)
    return total / max(1, len(y_idx))


def _argmax(row):
    best, bi = row[0], 0
    for i, v in enumerate(row):
        if v > best:
            best, bi = v, i
    return bi


def accuracy(proba, y_idx):
    hits = sum(1 for row, yi in zip(proba, y_idx) if _argmax(row) == yi)
    return hits / max(1, len(y_idx))


def balanced_accuracy(proba, y_idx):
    k = max((len(r) for r in proba), default=0)
    rec = []
    for c in range(k):
        idx = [i for i, yi in enumerate(y_idx) if yi == c]
        if not idx:
            continue
        rec.append(sum(1 for i in idx if _argmax(proba[i]) == c) / len(idx))
    return sum(rec) / max(1, len(rec))


def roc_auc(proba, y_idx):
    """Binary only: Mann-Whitney U on the classes[1] score, tie-aware."""
    if max((len(r) for r in proba), default=0) != 2:
        return None
    scores = sorted(((row[1], yi) for row, yi in zip(proba, y_idx)), key=lambda t: t[0])
    n_pos = sum(1 for _, yi in scores if yi == 1)
    n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and scores[j][0] == scores[i][0]:
            j += 1
        avg = (i + j + 1) / 2.0  # 1-based average rank of the tie group [i, j)
        for t in range(i, j):
            ranks[t] = avg
        i = j
    rank_sum_pos = sum(r for r, s in zip(ranks, scores) if s[1] == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def macro_f1(proba, y_idx):
    k = max((len(r) for r in proba), default=0)
    f1s = []
    for c in range(k):
        tp = sum(1 for row, yi in zip(proba, y_idx) if _argmax(row) == c and yi == c)
        fp = sum(1 for row, yi in zip(proba, y_idx) if _argmax(row) == c and yi != c)
        fn = sum(1 for row, yi in zip(proba, y_idx) if _argmax(row) != c and yi == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / max(1, k)


def score_cla(proba, y_true, classes):
    """Score one (model, dataset): proba rows aligned to `classes` order."""
    idx = {str(c): i for i, c in enumerate(classes)}
    y_idx = [idx[str(v)] for v in y_true]
    rows = expand_proba(proba)
    return {
        "logloss": log_loss(rows, y_idx),
        "accuracy": accuracy(rows, y_idx),
        "bal_acc": balanced_accuracy(rows, y_idx),
        "auc": roc_auc(rows, y_idx),
        "macro_f1": macro_f1(rows, y_idx),
    }
