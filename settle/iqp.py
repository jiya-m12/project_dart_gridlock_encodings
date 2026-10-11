"""E1: IQP encoding (De Santis, Tirone, Marmi, Giovannetti, arXiv:2406.07681, Sec. III A).

For one bank with d local payment bits y and k slack bits s, find a QUADRATIC
polynomial P(y, s) with
    violating y:  P(y, s) <= -1   for all s
    satisfying y: P(y, s) <=  0   for all s, and P(y, s) = 0 for at least one s
using the fewest slack bits k (k = 0, 1, 2, ... until feasible). Among those,
we pick the one with the smallest max |coefficient| (exact MILP, witness binaries).
Penalty added to the cost (minimisation): -lambda_u * P(y, s).

The polynomial depends only on WHICH local assignments are feasible (the
pattern), not on the amounts, so results are cached by (d, pattern).
Derivation needs scipy.optimize.milp; it is run offline and shipped as JSON.
"""
import json
from itertools import combinations

import numpy as np


def monomials(nv):
    """Constant, linear, pairwise monomials over nv binary variables."""
    return [()] + [(i,) for i in range(nv)] + list(combinations(range(nv), 2))


def _phi(assign_bits, nv, mons):
    return np.array([all((assign_bits >> i) & 1 for i in t) for t in mons], dtype=float)


def derive(d, feasible, k, coef_bound=8.0, time_limit=20.0, minimise=True):
    """feasible: bool list of length 2^d (index bit j = local payment j).
    Returns (coefs dict {monomial: value} in local indices (0..d-1 payments,
    d..d+k-1 slack), max|coef|) or None if no polynomial exists with k slack
    bits and |coef| <= coef_bound. Raises TimeoutError if undecided in time.
    minimise=False only tests feasibility (fast); True also minimises max|coef|."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    nv = d + k
    mons = monomials(nv)
    m = len(mons)
    feas_x = [x for x in range(2 ** d) if feasible[x]]
    ny = len(feas_x) * 2 ** k
    nvar = m + 1 + ny                      # [a (m), t, witnesses y]
    rows, lo, hi = [], [], []
    for x in range(2 ** d):
        for s in range(2 ** k):
            ph = _phi(x | (s << d), nv, mons)
            r = np.zeros(nvar); r[:m] = ph
            if feasible[x]:
                rows.append(r); lo.append(-np.inf); hi.append(0.0)          # P <= 0
                big_m = coef_bound * ph.sum() + 1.0                          # tight per-row big-M
                yi = m + 1 + feas_x.index(x) * 2 ** k + s
                r2 = r.copy(); r2[yi] = -big_m                               # P >= -M (1 - y)
                rows.append(r2); lo.append(-big_m); hi.append(np.inf)
            else:
                rows.append(r); lo.append(-np.inf); hi.append(-1.0)         # P <= -1
    for j in range(len(feas_x)):                                            # >= 1 witness per feasible x
        r = np.zeros(nvar); r[m + 1 + j * 2 ** k: m + 1 + (j + 1) * 2 ** k] = 1
        rows.append(r); lo.append(1.0); hi.append(np.inf)
    for i in range(m):                                                      # |a_i| <= t
        r = np.zeros(nvar); r[i] = 1; r[m] = -1; rows.append(r); lo.append(-np.inf); hi.append(0.0)
        r = np.zeros(nvar); r[i] = 1; r[m] = 1; rows.append(r); lo.append(0.0); hi.append(np.inf)
    c = np.zeros(nvar)
    if minimise:
        c[m] = 1.0
    integrality = np.zeros(nvar); integrality[m + 1:] = 1
    lb = np.concatenate([-coef_bound * np.ones(m), [0.0], np.zeros(ny)])
    ub = np.concatenate([coef_bound * np.ones(m), [coef_bound], np.ones(ny)])
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "disp": False})
    if res.status == 2:                    # proven infeasible
        return None
    if res.x is None:
        raise TimeoutError("IQP MILP undecided (d=%d, k=%d): %s" % (d, k, res.message))
    # Polish: big-M + integrality tolerances can leave P(witness) at ~ -1e-4.
    # Fix the witness choice and re-solve as a pure LP with exact equalities.
    y = res.x[m + 1:].reshape(len(feas_x), 2 ** k) if ny else np.zeros((0, 1))
    witness = {x: int(np.argmax(y[j])) for j, x in enumerate(feas_x)}
    return _polish(d, k, feasible, witness, coef_bound, minimise)


def _polish(d, k, feasible, witness, coef_bound, minimise):
    from scipy.optimize import linprog
    nv = d + k
    mons = monomials(nv)
    m = len(mons)
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for x in range(2 ** d):
        for s in range(2 ** k):
            ph = np.concatenate([_phi(x | (s << d), nv, mons), [0.0]])
            if feasible[x]:
                if s == witness[x]:
                    A_eq.append(ph); b_eq.append(0.0)
                else:
                    A_ub.append(ph); b_ub.append(0.0)
            else:
                A_ub.append(ph); b_ub.append(-1.0)
    for i in range(m):
        r = np.zeros(m + 1); r[i] = 1; r[m] = -1; A_ub.append(r); b_ub.append(0.0)
        r = np.zeros(m + 1); r[i] = -1; r[m] = -1; A_ub.append(r); b_ub.append(0.0)
    c = np.zeros(m + 1); c[m] = 1.0 if minimise else 0.0
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq) if A_eq else None,
                  b_eq=b_eq if b_eq else None, bounds=[(-coef_bound, coef_bound)] * m + [(0, coef_bound)],
                  method="highs")
    if res.status != 0:
        raise RuntimeError("IQP polish LP failed: " + res.message)
    a = res.x[:m]
    coefs = {mons[i]: float(a[i]) for i in range(m) if abs(a[i]) > 1e-10}
    return coefs, float(max(abs(v) for v in coefs.values()))


def check(d, k, feasible, coefs, tol=1e-7):
    nv = d + k
    for x in range(2 ** d):
        vals = []
        for s in range(2 ** k):
            b = x | (s << d)
            vals.append(sum(c for t, c in coefs.items() if all((b >> i) & 1 for i in t)))
        vals = np.array(vals)
        if feasible[x]:
            if abs(vals.max()) > tol:          # max_s P = 0
                return False
        elif vals.max() > -1 + tol:
            return False
    return True


def derive_min_slack(d, feasible, k_max=5, time_limit=120.0):
    """Smallest k with a valid polynomial (feasibility MILP), then minimise max|coef|
    at that k (keeps the best incumbent if the time limit hits)."""
    for k in range(k_max + 1):
        first = derive(d, feasible, k, time_limit=time_limit, minimise=False)
        if first is None:
            continue
        coefs, t, minimised = None, None, False
        for bound in (8.0, 16.0):       # HiGHS sometimes stalls at bound 8; a looser big-M helps
            try:
                coefs, t = derive(d, feasible, k, coef_bound=bound, time_limit=time_limit, minimise=True)
                minimised = True
                break
            except TimeoutError:
                pass
        if not minimised:               # keep the feasible (not coefficient-minimised) polynomial
            coefs, t = first
        assert check(d, k, feasible, coefs), "derived polynomial failed verification"
        return {"d": d, "k": k, "coefs": coefs, "max_coef": t, "minimised": minimised}
    raise RuntimeError("no IQP polynomial with <= %d slack bits" % k_max)


def pattern_key(d, feasible):
    return "%d:%s" % (d, "".join("1" if f else "0" for f in feasible))


def local_pattern(inst, u):
    vs, cs = inst.local_coeffs(u)
    d = len(vs)
    feas = [sum(cs[j] for j in range(d) if (m >> j) & 1) <= inst.liquidity[u] for m in range(2 ** d)]
    return vs, feas


def save_cache(cache, path):
    out = {key: {"d": v["d"], "k": v["k"], "max_coef": v["max_coef"], "minimised": v.get("minimised", True),
                 "coefs": [[list(t), c] for t, c in v["coefs"].items()]} for key, v in cache.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=0)


def load_cache_text(text):
    raw = json.loads(text)
    return {key: {"d": v["d"], "k": v["k"], "max_coef": v["max_coef"],
                  "minimised": v.get("minimised", True), "coefs": {tuple(t): c for t, c in v["coefs"]}} for key, v in raw.items()}
