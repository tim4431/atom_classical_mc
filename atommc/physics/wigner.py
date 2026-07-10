"""Wigner 3-j and 6-j symbols for hyperfine angular-momentum algebra.

Exact Racah formulas evaluated with integer factorials, supporting
half-integer angular momenta. Only intended for the small arguments of
hyperfine table construction (`physics.hyperfine`), where each symbol is
evaluated once at build time — no attempt is made to be fast or to avoid
overflow for large `j` (fine below `j ~ 20`).
"""

from __future__ import annotations

from math import factorial


def wigner_3j(
    j1: float, j2: float, j3: float, m1: float, m2: float, m3: float
) -> float:
    """Wigner 3-j symbol `(j1 j2 j3; m1 m2 m3)` (Racah formula)."""

    two_j = [_as_two(x, "j") for x in (j1, j2, j3)]
    two_m = [_as_two(x, "m") for x in (m1, m2, m3)]
    if any(tj < 0 for tj in two_j):
        raise ValueError("angular momenta must be non-negative.")
    if any((tj - tm) % 2 != 0 or abs(tm) > tj for tj, tm in zip(two_j, two_m)):
        return 0.0
    if sum(two_m) != 0:
        return 0.0
    if not _triangle(*two_j):
        return 0.0

    tj1, tj2, tj3 = two_j
    tm1, tm2, tm3 = two_m
    # All factorial arguments below are integers when the selection rules
    # above pass; `_half_int` asserts that.
    delta = (
        _fact2(tj1 + tj2 - tj3)
        * _fact2(tj1 - tj2 + tj3)
        * _fact2(-tj1 + tj2 + tj3)
        / _fact2(tj1 + tj2 + tj3 + 2)
    )
    norm = (
        _fact2(tj1 + tm1)
        * _fact2(tj1 - tm1)
        * _fact2(tj2 + tm2)
        * _fact2(tj2 - tm2)
        * _fact2(tj3 + tm3)
        * _fact2(tj3 - tm3)
    )

    k_min = max(0, _half_int(tj2 - tj3 - tm1), _half_int(tj1 - tj3 + tm2))
    k_max = min(
        _half_int(tj1 + tj2 - tj3),
        _half_int(tj1 - tm1),
        _half_int(tj2 + tm2),
    )
    total = 0.0
    for k in range(k_min, k_max + 1):
        denom = (
            factorial(k)
            * _fact2(tj1 + tj2 - tj3 - 2 * k)
            * _fact2(tj1 - tm1 - 2 * k)
            * _fact2(tj2 + tm2 - 2 * k)
            * _fact2(tj3 - tj2 + tm1 + 2 * k)
            * _fact2(tj3 - tj1 - tm2 + 2 * k)
        )
        total += (-1.0) ** k / denom
    sign = (-1.0) ** _half_int(tj1 - tj2 - tm3)
    return float(sign * (delta * norm) ** 0.5 * total)


def wigner_6j(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float
) -> float:
    """Wigner 6-j symbol `{j1 j2 j3; j4 j5 j6}` (Racah formula)."""

    tj = [_as_two(x, "j") for x in (j1, j2, j3, j4, j5, j6)]
    if any(x < 0 for x in tj):
        raise ValueError("angular momenta must be non-negative.")
    t1, t2, t3, t4, t5, t6 = tj
    triads = ((t1, t2, t3), (t1, t5, t6), (t4, t2, t6), (t4, t5, t3))
    if not all(_triangle(*tri) for tri in triads):
        return 0.0

    def tri_factor(ta: int, tb: int, tc: int) -> float:
        return (
            _fact2(ta + tb - tc)
            * _fact2(ta - tb + tc)
            * _fact2(-ta + tb + tc)
            / _fact2(ta + tb + tc + 2)
        )

    prefactor = 1.0
    for tri in triads:
        prefactor *= tri_factor(*tri)

    k_min = max(_half_int(sum(tri)) for tri in triads)
    k_max = min(
        _half_int(t1 + t2 + t4 + t5),
        _half_int(t2 + t3 + t5 + t6),
        _half_int(t3 + t1 + t6 + t4),
    )
    total = 0.0
    for k in range(k_min, k_max + 1):
        denom = (
            _fact2(2 * k - t1 - t2 - t3)
            * _fact2(2 * k - t1 - t5 - t6)
            * _fact2(2 * k - t4 - t2 - t6)
            * _fact2(2 * k - t4 - t5 - t3)
            * _fact2(t1 + t2 + t4 + t5 - 2 * k)
            * _fact2(t2 + t3 + t5 + t6 - 2 * k)
            * _fact2(t3 + t1 + t6 + t4 - 2 * k)
        )
        total += (-1.0) ** k * factorial(k + 1) / denom
    return float(prefactor**0.5 * total)


def _as_two(x: float, kind: str) -> int:
    """Return `2x` as an exact integer (angular momenta are half-integer)."""

    two = round(2.0 * float(x))
    if abs(2.0 * float(x) - two) > 1.0e-9:
        raise ValueError(f"{kind} = {x} is not integer or half-integer.")
    return int(two)


def _half_int(two_x: int) -> int:
    """Halve a doubled quantity that must be an even integer."""

    if two_x % 2 != 0:
        raise ValueError("internal error: expected an integer-valued quantity.")
    return two_x // 2


def _fact2(two_x: int) -> int:
    """Factorial of `two_x / 2`, which must be an even non-negative integer."""

    return factorial(_half_int(two_x))


def _triangle(ta: int, tb: int, tc: int) -> bool:
    """Triangle rule on doubled angular momenta, including parity."""

    return abs(ta - tb) <= tc <= ta + tb and (ta + tb + tc) % 2 == 0
