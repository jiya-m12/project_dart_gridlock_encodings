"""Pseudo-Boolean polynomials over binary variables x_i in {0,1}.

A polynomial is a dict {tuple(sorted variable indices): coefficient}; the empty
tuple is the constant. All encodings produce one of these (to be MINIMISED);
QAOA uses its Ising form via x_i = (1 - Z_i) / 2.
"""
from itertools import combinations

import numpy as np


def add(p, q, scale=1.0):
    out = dict(p)
    for t, c in q.items():
        out[t] = out.get(t, 0.0) + scale * c
    return out


def clean(p, tol=1e-12):
    return {t: c for t, c in p.items() if abs(c) > tol}


def scale(p, s):
    return {t: s * c for t, c in p.items()}


def degree(p):
    return max((len(t) for t, c in p.items() if abs(c) > 1e-12), default=0)


def variables(p):
    return sorted({v for t in p for v in t})


def square_of_linear(lin, const=0.0):
    """(const + sum_i lin[i] x_i)^2 expanded using x_i^2 = x_i."""
    p = {(): const * const}
    items = sorted(lin.items())
    for i, a in items:
        p[(i,)] = p.get((i,), 0.0) + a * a + 2 * const * a
    for (i, a), (j, b) in combinations(items, 2):
        p[(i, j)] = p.get((i, j), 0.0) + 2 * a * b
    return clean(p)


def linear(lin, const=0.0):
    p = {(): const}
    for i, a in lin.items():
        p[(i,)] = p.get((i,), 0.0) + a
    return clean(p)


def from_table(vars_, table):
    """Unique multilinear polynomial equal to `table` on {0,1}^d.
    table[m] is the value when x[vars_[j]] = (m >> j) & 1. Moebius inversion:
    c_S = sum_{T subset S} (-1)^{|S|-|T|} f(T)."""
    d = len(vars_)
    c = np.array(table, dtype=float).copy()
    assert c.shape == (2 ** d,)
    for j in range(d):           # fast Moebius transform
        bit = 1 << j
        for m in range(2 ** d):
            if m & bit:
                c[m] -= c[m ^ bit]
    p = {}
    for m in range(2 ** d):
        if abs(c[m]) > 1e-12:
            p[tuple(sorted(vars_[j] for j in range(d) if (m >> j) & 1))] = float(c[m])
    return p


def evaluate_all(p, n, chunk_bits=20):
    """Values of p on all 2^n bitstrings (index bit j = variable j)."""
    out = np.empty(2 ** n)
    step = 2 ** min(n, chunk_bits)
    for start in range(0, 2 ** n, step):
        idx = np.arange(start, start + step, dtype=np.int64)
        v = np.zeros(step)
        for t, c in p.items():
            if not t:
                v += c
                continue
            mask = 0
            for i in t:
                mask |= 1 << i
            v += c * ((idx & mask) == mask)
        out[start:start + step] = v
    return out


def to_ising(p):
    """Binary polynomial -> Z polynomial {tuple: coef} with x = (1 - Z)/2."""
    z = {}
    for t, c in p.items():
        k = len(t)
        for r in range(k + 1):
            for sub in combinations(t, r):
                z[sub] = z.get(sub, 0.0) + c * ((-1) ** r) / (2 ** k)
    return clean(z)


def term_stats(zpoly):
    """Counts of non-constant Ising terms by order, and CNOTs for a CNOT-ladder
    implementation (2(k-1) per k-body term, no cancellation between terms)."""
    by_order = {}
    cnot = 0
    for t in zpoly:
        if len(t) == 0:
            continue
        by_order[len(t)] = by_order.get(len(t), 0) + 1
        cnot += 2 * (len(t) - 1)
    return {"terms_by_order": dict(sorted(by_order.items())), "cnot_ladder": cnot,
            "max_order": max(by_order, default=0)}
