"""Exactness checks: does minimising the encoded cost give the true optimum?"""
import numpy as np

from . import poly as P
from .exact import all_bitstrings, feasibility_mask, values, is_value


def reduced_cost(enc):
    """f(x) = min over slack bits of cost(x, s), for all 2^n payment selections.
    Uses the per-bank factorisation (slack bits of different banks are disjoint)."""
    n = enc.n_logical
    idx = np.arange(2 ** n, dtype=np.int64)
    f = P.evaluate_all(enc.objective, n)
    for b in enc.blocks:
        loc = b.xvars + b.svars
        remap = {v: i for i, v in enumerate(loc)}
        lp = {tuple(sorted(remap[v] for v in t)): c for t, c in b.poly.items()}
        vals = P.evaluate_all(lp, len(loc))                       # index: x bits low, s bits high
        tab = vals.reshape(2 ** len(b.svars), 2 ** len(b.xvars)).min(axis=0)
        local_idx = np.zeros_like(idx)
        for j, v in enumerate(b.xvars):
            local_idx |= ((idx >> v) & 1) << j
        f = f + tab[local_idx]
    return f


def check_exact(inst, enc, tol=1e-7):
    """Returns dict: exact (all minimisers of f are feasible optima), plus details."""
    f = reduced_cost(enc)
    feas = feasibility_mask(inst)
    v = values(inst)
    opt = v[feas].max()
    fmin = f.min()
    argmins = np.flatnonzero(f <= fmin + tol)
    good = feas[argmins] & is_value(v[argmins], opt)
    return {"exact": bool(good.all()), "opt": float(opt), "fmin": float(fmin),
            "n_argmin": int(len(argmins)),
            "argmin_feasible": bool(feas[argmins].all()),
            "argmin_value": float(v[argmins[0]])}


def check_full_enumeration(enc, inst, max_qubits=20):
    """Independent check on small encodings: enumerate all 2^(n_total) states
    of the full polynomial, project to payment bits."""
    if enc.n_total > max_qubits:
        return None
    full = P.evaluate_all(enc.poly, enc.n_total)
    n = enc.n_logical
    f = full.reshape(-1, 2 ** n).min(axis=0)
    return f
