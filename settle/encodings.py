"""Constraint encodings for liquidity-only settlement.

Every encoding returns an Encoding: cost polynomial to MINIMISE over
logical bits x_0..x_{n-1} (payments) plus auxiliary/slack bits n.. .
    cost(x, s) = -sum_k w_k x_k + sum_u penalty_u(x_local(u), s_u)
Slack bits of different banks are disjoint, which lets us check exactness
without enumerating slack: min over s factorises per bank.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import poly as P


@dataclass
class Block:
    bank: int
    xvars: list      # global indices of the payments in this bank's constraint
    svars: list      # global indices of this bank's slack bits
    poly: dict       # penalty polynomial in global variable indices


@dataclass
class Encoding:
    name: str
    n_logical: int
    n_total: int
    objective: dict
    blocks: list
    params: dict = field(default_factory=dict)

    @property
    def poly(self):
        p = dict(self.objective)
        for b in self.blocks:
            p = P.add(p, b.poly)
        return P.clean(p)

    @property
    def n_slack(self):
        return self.n_total - self.n_logical


# ---------------------------------------------------------------- penalty weights
def penalty_weights(inst, rule="global", gamma=2.0):
    """lambda_u per bank.
    local:     gamma * total amount of payments incident to u
    neighbour: gamma * total amount of payments incident to u or to any bank adjacent to u
               (each payment counted once; NB differs from De Santis et al. Eq. 68)
    global:    gamma * total amount of all payments (gamma > 1 guarantees exactness
               for E0/E1/E2/E4: any violation costs >= lambda > sum w)"""
    W = inst.weights
    lam = []
    for u in range(inst.n_banks):
        if rule == "local":
            arcs = set(inst.incident(u))
        elif rule == "neighbour":
            nb = {u}
            for k in inst.incident(u):
                nb |= {inst.arcs[k][0], inst.arcs[k][1]}
            arcs = set()
            for v in nb:
                arcs |= set(inst.incident(v))
        elif rule == "global":
            arcs = set(range(inst.n))
        else:
            raise ValueError(rule)
        lam.append(gamma * float(sum(W[k] for k in arcs)))
    return lam


def binding_banks(inst):
    """Banks whose constraint can actually be violated (max net outflow > C_u)."""
    out = []
    for u in range(inst.n_banks):
        vs, cs = inst.local_coeffs(u)
        if sum(c for c in cs if c > 0) > inst.liquidity[u]:
            out.append(u)
    return out


def objective_poly(inst):
    return {(k,): -float(w) for k, w in enumerate(inst.weights)}


# ---------------------------------------------------------------- E0 standard slack
def e0_standard(inst, rule="global", gamma=2.0):
    """net_out_u + s_u = C_u with binary slack s_u = sum_j 2^j b_j,
    k_u = ceil(log2(C_u + sum_in_u + 1)) bits; penalty lambda_u (net_out_u + s_u - C_u)^2."""
    lam = penalty_weights(inst, rule, gamma)
    nxt = inst.n
    blocks = []
    for u in binding_banks(inst):
        vs, cs = inst.local_coeffs(u)
        s_max = inst.liquidity[u] + sum(-c for c in cs if c < 0)
        k = max(1, math.ceil(math.log2(s_max + 1)))
        svars = list(range(nxt, nxt + k))
        nxt += k
        lin = {v: float(c) for v, c in zip(vs, cs)}
        for j, sv in enumerate(svars):
            lin[sv] = float(2 ** j)
        pen = P.scale(P.square_of_linear(lin, -float(inst.liquidity[u])), lam[u])
        blocks.append(Block(u, vs, svars, pen))
    return Encoding("E0_standard", inst.n, nxt, objective_poly(inst), blocks,
                    {"rule": rule, "gamma": gamma, "lambda": lam})


# ---------------------------------------------------------------- slack-free, from a local function
def _local_table(inst, u, fn):
    """fn(g) evaluated on every local assignment, g = net_out_u - C_u."""
    vs, cs = inst.local_coeffs(u)
    d = len(vs)
    tab = np.empty(2 ** d)
    for m in range(2 ** d):
        net = sum(cs[j] for j in range(d) if (m >> j) & 1)
        tab[m] = fn(net - inst.liquidity[u])
    return vs, tab


def e_local_function(inst, name, fn, rule="global", gamma=2.0):
    """Penalty lambda_u * fn(g_u) expanded exactly into a (higher-order) polynomial
    on the bank's own payment bits. No slack. Order <= bank degree."""
    lam = penalty_weights(inst, rule, gamma)
    blocks = []
    for u in binding_banks(inst):
        vs, tab = _local_table(inst, u, fn)
        blocks.append(Block(u, vs, [], P.scale(P.from_table(vs, tab), lam[u])))
    return Encoding(name, inst.n, inst.n, objective_poly(inst), blocks,
                    {"rule": rule, "gamma": gamma, "lambda": lam})


def e2_indicator(inst, rule="global", gamma=2.0):
    """E2: lambda_u * 1[g_u > 0]  (exact step penalty, higher-order)."""
    return e_local_function(inst, "E2_indicator", lambda g: 1.0 if g > 0 else 0.0, rule, gamma)


def e4_overflow_sq(inst, rule="global", gamma=2.0):
    """E4: lambda_u * max(0, g_u)^2  (graded overflow penalty, higher-order).
    With integer amounts the smallest violation costs lambda_u, so the same
    weight rule as E2 keeps it exact."""
    return e_local_function(inst, "E4_overflow_sq", lambda g: max(0.0, g) ** 2, rule, gamma)


# ---------------------------------------------------------------- E3 unbalanced penalization
def e3_unbalanced(inst, l1, l2):
    """Montanez-Barrera et al. (arXiv:2211.13914): with h_u = C_u - net_out_u
    (constraint h_u >= 0), penalty  -l1 * h_u + l2 * h_u^2  in raw amount units.
    Quadratic, no slack, approximate (optimum not guaranteed); l1, l2 must be
    tuned on separate instances, never with knowledge of the optimum."""
    blocks = []
    for u in binding_banks(inst):
        vs, cs = inst.local_coeffs(u)
        lin = {v: -float(c) for v, c in zip(vs, cs)}
        h0 = float(inst.liquidity[u])
        pen = P.add(P.scale(P.linear(lin, h0), -l1), P.square_of_linear(lin, h0), l2)
        blocks.append(Block(u, vs, [], P.clean(pen)))
    return Encoding("E3_unbalanced", inst.n, inst.n, objective_poly(inst), blocks,
                    {"l1": l1, "l2": l2})


# ---------------------------------------------------------------- E1 IQP
def e1_iqp(inst, rule="global", gamma=2.0, cache=None, derive_missing=True):
    """IQP (De Santis et al.): per bank a quadratic P(y, s) with the fewest slack
    bits; penalty -lambda_u * P. `cache` maps pattern keys to derived polynomials
    (see settle/iqp.py); missing patterns are derived with scipy MILP if allowed."""
    from . import iqp
    cache = {} if cache is None else cache
    lam = penalty_weights(inst, rule, gamma)
    nxt = inst.n
    blocks = []
    for u in binding_banks(inst):
        vs, feas = iqp.local_pattern(inst, u)
        key = iqp.pattern_key(len(vs), feas)
        if key not in cache:
            if not derive_missing:
                raise KeyError("IQP polynomial for pattern %s not precomputed" % key)
            cache[key] = iqp.derive_min_slack(len(vs), feas)
        r = cache[key]
        d, k = r["d"], r["k"]
        svars = list(range(nxt, nxt + k))
        nxt += k
        gl = vs + svars
        pen = {}
        for t, c in r["coefs"].items():
            gt = tuple(sorted(gl[i] for i in t))
            pen[gt] = pen.get(gt, 0.0) - lam[u] * c
        blocks.append(Block(u, vs, svars, P.clean(pen)))
    return Encoding("E1_iqp", inst.n, nxt, objective_poly(inst), blocks,
                    {"rule": rule, "gamma": gamma, "lambda": lam})
