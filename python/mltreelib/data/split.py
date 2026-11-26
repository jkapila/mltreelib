"""Deterministic row splitting: train/test, train/validation/test, k-fold and take.

Every splitter returns row index lists that refer to the input rows. The result
depends only on ``(n, the arguments, seed)``. It never depends on the global
``random`` state, the platform, or ``PYTHONHASHSEED``, because:

- randomness comes only from ``random.Random(seed)``, and only its ``random()``
  method is used (a Mersenne Twister output with a specified bit layout), through
  our own Fisher-Yates shuffle, so no version-specific ``shuffle``/``sample``
  logic is involved;
- labels and groups are ordered by first appearance, never by hash or set order;
- ties are broken by row index.

Missing labels (``None`` or NaN) in ``stratify`` and ``groups`` are kept as their own
level or group, labelled ``None``. Missing ``time_order`` keys sort after all
observed keys. No row is ever dropped.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from mltreelib.data._common import is_missing

__all__ = [
    "kfold",
    "repeated_kfold",
    "take",
    "train_test_split",
    "train_val_test_split",
]


# ---------------------------------------------------------------------------
# Validation and normalisation
# ---------------------------------------------------------------------------


def _check_seed(seed: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")


def _check_n(n: Any) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")


def _check_splittable(n: int) -> None:
    if n < 2:
        raise ValueError(f"need at least 2 rows to split, got n={n}")


def _labels(values: Any, n: int, name: str) -> list[Any] | None:
    """Return a copy with missing entries replaced by None, or None if not given."""
    if values is None:
        return None
    out = [None if is_missing(v) else v for v in values]
    if len(out) != n:
        raise ValueError(f"{name} has {len(out)} entries, expected n={n}")
    return out


def _normalise(
    n: int, stratify: Any, groups: Any, time_order: Any
) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
    """Validate the mode arguments. ``groups`` may combine with ``stratify``."""
    if time_order is not None and (stratify is not None or groups is not None):
        raise ValueError("time_order cannot be combined with stratify or groups")
    return (
        _labels(stratify, n, "stratify"),
        _labels(groups, n, "groups"),
        _labels(time_order, n, "time_order"),
    )


def _resolve_size(n: int, size: Any, name: str) -> int:
    """Turn a fraction (float in (0, 1)) or absolute count (int >= 1) into a row count."""
    if isinstance(size, bool):
        raise TypeError(f"{name} must be a float or an int, not bool")
    if isinstance(size, float):
        if not 0.0 < size < 1.0:
            raise ValueError(f"{name} as a float must be in (0, 1), got {size!r}")
        # Decimal(repr(x)) gives the decimal the caller wrote, so 0.25 * 10 rounds half up to 3.
        count = int((Decimal(repr(size)) * n).to_integral_value(rounding=ROUND_HALF_UP))
        return min(max(count, 1), n - 1)
    if isinstance(size, int):
        if not 1 <= size <= n - 1:
            raise ValueError(f"{name}={size} must be between 1 and n-1={n - 1}")
        return size
    raise TypeError(f"{name} must be a float or an int, got {type(size).__name__}")


# ---------------------------------------------------------------------------
# Deterministic primitives
# ---------------------------------------------------------------------------


def _shuffle(items: Sequence[int], rng: random.Random) -> list[int]:
    """Fisher-Yates shuffle driven only by ``rng.random()``."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        out[i], out[j] = out[j], out[i]
    return out


def _first_appearance(labels: Sequence[Any]) -> dict[Any, list[int]]:
    """Map each label to its row positions. Dict order is first appearance."""
    out: dict[Any, list[int]] = {}
    for pos, label in enumerate(labels):
        out.setdefault(label, []).append(pos)
    return out


def _chunks(seq: Sequence[int], m: int) -> list[list[int]]:
    """Cut ``seq`` into ``m`` consecutive blocks; the first ``len % m`` blocks get one extra."""
    base, extra = divmod(len(seq), m)
    out: list[list[int]] = []
    start = 0
    for j in range(m):
        size = base + (1 if j < extra else 0)
        out.append(list(seq[start : start + size]))
        start += size
    return out


def _time_order(keys: Sequence[Any]) -> list[int]:
    """Row positions sorted by (key, index). Missing keys sort last."""

    def sort_key(pos: int) -> tuple[int, Any, int]:
        value = keys[pos]
        return (1, 0, pos) if value is None else (0, value, pos)

    try:
        return sorted(range(len(keys)), key=sort_key)
    except TypeError as exc:
        raise ValueError("time_order keys must be mutually comparable") from exc


def _allocate(counts: Sequence[int], total: int) -> list[int]:
    """Largest-remainder split of ``total`` across classes of the given sizes.

    Each class gets floor or ceil of its exact share, so class proportions match
    within one row. A class keeps at least one row on the train side when possible:
    the caps tighten in tiers only if the requested total cannot fit otherwise.
    Ties in remainders go to the earlier class.
    """
    n = sum(counts)
    floors = [(c * total) // n for c in counts]
    rems = [(c * total) % n for c in counts]
    cap_tiers = [
        [c - 1 for c in counts],  # every class keeps one row in train
        [c if c >= 2 else 0 for c in counts],  # singletons still stay in train
        list(counts),  # last resort: any class may go entirely to test
    ]
    caps = next(tier for tier in cap_tiers if sum(tier) >= total)
    base = [min(f, cap) for f, cap in zip(floors, caps)]
    left = total - sum(base)
    order = sorted(range(len(counts)), key=lambda i: (-rems[i], i))
    while left > 0:
        for i in order:
            if left == 0:
                break
            if base[i] < caps[i]:
                base[i] += 1
                left -= 1
    return base


def _stratified_test(
    strat: Sequence[Any], test_count: int, rng: random.Random
) -> list[int]:
    classes = list(_first_appearance(strat).values())
    alloc = _allocate([len(rows) for rows in classes], test_count)
    test: list[int] = []
    for rows, take_n in zip(classes, alloc):
        test.extend(_shuffle(rows, rng)[:take_n])
    return test


def _group_test(
    groups: Sequence[Any],
    strat: Sequence[Any] | None,
    test_count: int,
    rng: random.Random,
) -> list[int]:
    """Whole groups to test until the test count reaches ``test_count``.

    Without ``strat`` the groups are taken in seeded-shuffle order. With ``strat``
    each group has a majority label (ties go to the label seen first in the group),
    and the next group is taken from the label furthest below its target share.
    This is approximate: a group can overshoot the target by its size.
    """
    group_rows = _first_appearance(groups)
    gids = list(group_rows)
    ordered = [gids[j] for j in _shuffle(range(len(gids)), rng)]
    test_groups: list[Any] = []
    count = 0

    if strat is None:
        for gid in ordered:
            if count >= test_count:
                break
            test_groups.append(gid)
            count += len(group_rows[gid])
    else:
        n = len(strat)
        class_rows = {
            label: len(rows) for label, rows in _first_appearance(strat).items()
        }
        queues: dict[Any, deque[Any]] = {label: deque() for label in class_rows}
        for gid in ordered:
            tally: dict[Any, int] = {}
            for pos in group_rows[gid]:
                tally[strat[pos]] = tally.get(strat[pos], 0) + 1
            majority = max(tally.items(), key=lambda kv: kv[1])[0]
            queues[majority].append(gid)
        test_per_class = {label: 0 for label in class_rows}
        while count < test_count:
            best_label = None
            best_deficit = None
            for label, size in class_rows.items():  # first-appearance order breaks ties
                if not queues[label]:
                    continue
                # Scaled by n to stay in integers: size/n * test_count - test_per_class.
                deficit = size * test_count - test_per_class[label] * n
                if best_deficit is None or deficit > best_deficit:
                    best_label, best_deficit = label, deficit
            if best_label is None:
                break
            gid = queues[best_label].popleft()
            test_groups.append(gid)
            count += len(group_rows[gid])
            for pos in group_rows[gid]:
                test_per_class[strat[pos]] += 1

    return [pos for gid in test_groups for pos in group_rows[gid]]


def _split(
    n: int,
    test_count: int,
    seed: int,
    strat: list[Any] | None,
    groups: list[Any] | None,
    time: list[Any] | None,
) -> tuple[list[int], list[int]]:
    """Return sorted (train, test) positions for one split. Inputs are pre-validated."""
    rng = random.Random(seed)
    if time is not None:
        order = _time_order(time)
        cut = n - test_count
        return sorted(order[:cut]), sorted(order[cut:])
    if groups is not None:
        test = _group_test(groups, strat, test_count, rng)
    elif strat is not None:
        test = _stratified_test(strat, test_count, rng)
    else:
        test = _shuffle(range(n), rng)[:test_count]
    test_sorted = sorted(test)
    test_set = set(test_sorted)
    train = [pos for pos in range(n) if pos not in test_set]
    if not test_sorted or not train:
        raise ValueError(
            "split leaves one side empty; with groups, a single group may hold most rows"
        )
    return train, test_sorted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_test_split(
    n: int,
    test_size: float = 0.2,
    seed: int = 0,
    stratify: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
    time_order: Sequence[Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Split ``n`` rows into train and test index lists.

    Modes (at most one of ``stratify``, ``groups``, ``time_order``; ``groups``
    may be combined with ``stratify``):

    - plain: a seeded shuffle, the first ``test`` rows go to test.
    - ``stratify``: each label's rows are split proportionally (largest remainder),
      so class counts match their exact share within one row. A class of one row
      goes to train.
    - ``groups``: whole groups go to test in seeded-shuffle order until the test
      count reaches the target. No group is split across the two sides.
    - ``groups`` with ``stratify``: a group's label is its majority label. Groups are
      taken from the label furthest below its target share. This is approximate.
    - ``time_order``: rows sorted by (key, index); the last ``test`` rows are test.
      No shuffling.

    Determinism: the result depends only on ``(n, arguments, seed)``. See the
    module docstring.

    Args:
        n: Number of rows. Must be at least 2.
        test_size: Float in (0, 1) for a fraction, rounded half up, then clamped to
            ``[1, n-1]``. Or an int >= 1 for an absolute count, which must be at
            most ``n-1``.
        seed: Integer seed for the shuffle.
        stratify: One label per row. ``None``/NaN is its own level.
        groups: One group id per row. ``None``/NaN is its own group.
        time_order: One sortable key per row. Missing keys sort last.

    Returns:
        ``(train_idx, test_idx)``, both sorted ascending, both non-empty, and
        together they cover ``range(n)`` exactly once.

    Raises:
        ValueError: For an invalid size, an invalid mode combination, a length
            mismatch, or a groups split that leaves one side empty.

    Example:
        >>> train_test_split(10, test_size=0.2, seed=0)
        ([0, 1, 2, 3, 5, 6, 7, 8], [4, 9])
    """
    _check_n(n)
    _check_seed(seed)
    _check_splittable(n)
    strat, grp, time = _normalise(n, stratify, groups, time_order)
    test_count = _resolve_size(n, test_size, "test_size")
    return _split(n, test_count, seed, strat, grp, time)


def train_val_test_split(
    n: int,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 0,
    stratify: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
    time_order: Sequence[Any] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Split ``n`` rows into train, validation and test index lists.

    The test split is made first with :func:`train_test_split`. The validation split
    is then made from the remaining rows with the same mode and seed. A float
    ``val_size`` is a fraction of the original ``n``, not of the remainder.

    Args:
        n: Number of rows. Must be at least 2.
        val_size: Fraction (float) of the original ``n``, or an absolute count (int).
        test_size: As in :func:`train_test_split`.
        seed: Integer seed, used for both splits.
        stratify: As in :func:`train_test_split`.
        groups: As in :func:`train_test_split`.
        time_order: As in :func:`train_test_split`. The validation rows are the
            latest rows of the remainder.

    Returns:
        ``(train_idx, val_idx, test_idx)``, each sorted ascending, together a
        partition of ``range(n)``.

    Raises:
        ValueError: If the validation count leaves no training rows, or for any
            invalid argument covered by :func:`train_test_split`.

    Example:
        >>> train_val_test_split(10, val_size=0.2, test_size=0.2, seed=0)
        ([1, 2, 5, 6, 7, 8], [0, 3], [4, 9])
    """
    _check_n(n)
    _check_seed(seed)
    _check_splittable(n)
    strat, grp, time = _normalise(n, stratify, groups, time_order)
    test_count = _resolve_size(n, test_size, "test_size")
    val_count = _resolve_size(n, val_size, "val_size")
    rest, test = _split(n, test_count, seed, strat, grp, time)
    m = len(rest)
    if val_count > m - 1:
        raise ValueError(
            f"val count {val_count} leaves no training rows: {m} rows remain after the test split"
        )

    def sub(values: list[Any] | None) -> list[Any] | None:
        return None if values is None else [values[pos] for pos in rest]

    train_pos, val_pos = _split(m, val_count, seed, sub(strat), sub(grp), sub(time))
    train = sorted(rest[pos] for pos in train_pos)
    val = sorted(rest[pos] for pos in val_pos)
    return train, val, test


def kfold(
    n: int,
    k: int = 5,
    seed: int = 0,
    stratify: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
    time_order: Sequence[Any] | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Split ``n`` rows into ``k`` (train, test) folds.

    Modes:

    - plain: seeded shuffle, then ``k`` consecutive blocks whose sizes differ by at
      most one.
    - ``stratify``: each class is shuffled within itself, then dealt round-robin
      across folds. The round-robin pointer carries over between classes, so fold
      sizes also differ by at most one. A class smaller than ``k`` is spread as
      evenly as possible and does not raise.
    - ``groups``: groups are taken in seeded-shuffle order and each goes to the
      currently smallest fold (ties to the lower fold index). No group appears in
      two folds. Requires at least ``k`` groups.
    - ``time_order``: expanding window. Rows are sorted by (key, index) and cut into
      ``k+1`` consecutive blocks. Fold ``i`` trains on blocks ``0..i`` and tests on
      block ``i+1``. Block 0 is never tested. Requires ``n >= k+1``.

    ``groups`` and ``stratify`` cannot be combined in ``kfold``. Each test fold is
    disjoint from the others, and the test folds cover all rows (except block 0 in
    time mode).

    Determinism: the result depends only on ``(n, arguments, seed)``. See the
    module docstring.

    Args:
        n: Number of rows.
        k: Number of folds, at least 2 and at most ``n``.
        seed: Integer seed.
        stratify: One label per row. ``None``/NaN is its own level.
        groups: One group id per row. ``None``/NaN is its own group.
        time_order: One sortable key per row. Missing keys sort last.

    Returns:
        A list of ``k`` ``(train_idx, test_idx)`` tuples, each list sorted ascending.

    Raises:
        ValueError: If ``k < 2``, ``k > n``, there are fewer than ``k`` groups, time
            mode has ``n < k+1``, or ``groups`` and ``stratify`` are both given.

    Example:
        >>> folds = kfold(6, k=3, seed=0)
        >>> sorted(i for _, test in folds for i in test)
        [0, 1, 2, 3, 4, 5]
        >>> [len(test) for _, test in folds]
        [2, 2, 2]
    """
    _check_n(n)
    _check_seed(seed)
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}; every fold needs at least one test row")
    strat, grp, time = _normalise(n, stratify, groups, time_order)
    if strat is not None and grp is not None:
        raise ValueError(
            "kfold does not combine groups with stratify; pass one of them"
        )

    if time is not None:
        if n < k + 1:
            raise ValueError(f"time_order kfold needs n >= k+1 = {k + 1}, got n={n}")
        blocks = _chunks(_time_order(time), k + 1)
        return [
            (
                sorted(pos for block in blocks[: i + 1] for pos in block),
                sorted(blocks[i + 1]),
            )
            for i in range(k)
        ]

    rng = random.Random(seed)
    tests: list[list[int]]
    if grp is not None:
        group_rows = _first_appearance(grp)
        if len(group_rows) < k:
            raise ValueError(
                f"groups kfold needs at least k={k} groups, got {len(group_rows)}"
            )
        folds: list[list[int]] = [[] for _ in range(k)]
        sizes = [0] * k
        row_lists = list(group_rows.values())
        for g in _shuffle(range(len(row_lists)), rng):
            rows = row_lists[g]
            target = min(range(k), key=lambda j: (sizes[j], j))
            folds[target].extend(rows)
            sizes[target] += len(rows)
        tests = folds
    elif strat is not None:
        tests = [[] for _ in range(k)]
        pointer = 0
        for rows in _first_appearance(strat).values():
            for pos in _shuffle(rows, rng):
                tests[pointer].append(pos)
                pointer = (pointer + 1) % k
    else:
        tests = _chunks(_shuffle(range(n), rng), k)

    out: list[tuple[list[int], list[int]]] = []
    for test in tests:
        test_sorted = sorted(test)
        test_set = set(test_sorted)
        out.append(([pos for pos in range(n) if pos not in test_set], test_sorted))
    return out


def repeated_kfold(
    n: int,
    k: int = 5,
    repeats: int = 3,
    seed: int = 0,
    stratify: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
) -> list[list[tuple[list[int], list[int]]]]:
    """Run :func:`kfold` ``repeats`` times, with repeat ``r`` using seed ``seed + r``.

    Args:
        n: Number of rows.
        k: Folds per repeat, as in :func:`kfold`.
        repeats: Number of repeats, at least 1.
        seed: Base seed. Repeat ``r`` uses ``seed + r``.
        stratify: As in :func:`kfold`.
        groups: As in :func:`kfold`. ``time_order`` is not offered here.

    Returns:
        A list of length ``repeats``. Each item is the :func:`kfold` result for that
        repeat.

    Raises:
        ValueError: If ``repeats < 1`` or any :func:`kfold` argument is invalid.

    Example:
        >>> runs = repeated_kfold(6, k=3, repeats=2, seed=0)
        >>> len(runs), len(runs[0])
        (2, 3)
    """
    _check_seed(seed)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"repeats must be an int >= 1, got {repeats!r}")
    return [kfold(n, k, seed + r, stratify, groups) for r in range(repeats)]


def take(columns: Any, indices: Sequence[int]) -> Any:
    """Subset rows of a dict of columns (or of a list) by index, keeping ``indices`` order.

    Values are copied by reference and never transformed, so missing values stay
    missing. The input is not modified.

    Args:
        columns: A dict mapping column name to a sequence of values, or a list or
            tuple of values.
        indices: Row positions to keep. Repeats are allowed. Order is preserved.

    Returns:
        A dict with the same keys (dict input) or a list or tuple of the same kind
        (sequence input), each holding the selected rows.

    Raises:
        IndexError: If an index is negative or at least the length of a column. The
            message names the bad index.
        TypeError: If an index is not an integer, or ``columns`` is not a dict,
            list or tuple.

    Example:
        >>> take({"a": [1, None, 3], "b": ["x", "y", "z"]}, [2, 0])
        {'a': [3, 1], 'b': ['z', 'x']}
        >>> take([10, 20, 30], [1])
        [20]
    """
    idx = list(indices)

    def pick(values: Sequence[Any], name: str) -> list[Any]:
        size = len(values)
        picked: list[Any] = []
        for raw in idx:
            if isinstance(raw, bool):
                raise TypeError("indices must be integers, not bool")
            try:
                i = raw.__index__()
            except AttributeError:
                raise TypeError(
                    f"indices must be integers, got {type(raw).__name__}"
                ) from None
            if not 0 <= i < size:
                raise IndexError(f"index {i} out of range for {name} with {size} rows")
            picked.append(values[i])
        return picked

    if isinstance(columns, Mapping):
        return {
            name: pick(values, f"column {name!r}") for name, values in columns.items()
        }
    if isinstance(columns, tuple):
        return tuple(pick(columns, "sequence"))
    if isinstance(columns, list):
        return pick(columns, "sequence")
    raise TypeError(
        f"columns must be a dict, list or tuple, got {type(columns).__name__}"
    )
