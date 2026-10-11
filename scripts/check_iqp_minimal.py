"""Check that every IQP polynomial in the shipped cache uses the fewest slack bits.
Independent of settle/iqp.py (own LP/MILP formulation).

k = 0 is a pure LP (no slack, so no witness variables): P(y) = 0 on satisfying y and
P(y) <= -1 on violating y, with UNBOUNDED coefficients. So "k = 0 impossible" is an
exact statement, not tied to a coefficient bound.
k = 1 needs witness binaries (which slack value makes P = 0). We test it with a MILP at
coefficient bounds 64 and 256; "impossible" here is relative to that bound.
Output: results/iqp_minimality.json."""
import os, sys, json
from itertools import combinations
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

PATH = "settle/iqp_data/iqp_cache.json"


def mons(nv):
    return [()] + [(i,) for i in range(nv)] + list(combinations(range(nv), 2))


def phi(z, nv, M):
    return np.array([float(all((z >> i) & 1 for i in t)) for t in M])


def k0_possible(d, feas):
    M = mons(d)
    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    for y in range(2 ** d):
        r = phi(y, d, M)
        if feas[y]:
            A_eq.append(r); b_eq.append(0.0)
        else:
            A_ub.append(r); b_ub.append(-1.0)
    res = linprog(np.zeros(len(M)), A_ub=np.array(A_ub) if A_ub else None, b_ub=b_ub or None,
                  A_eq=np.array(A_eq) if A_eq else None, b_eq=b_eq or None,
                  bounds=[(None, None)] * len(M), method="highs")
    return res.status == 0


def k1_possible(d, feas, B):
    nv = d + 1
    M = mons(nv); m = len(M)
    fx = [y for y in range(2 ** d) if feas[y]]
    nvar = m + 2 * len(fx)
    rows, lo, hi = [], [], []
    for y in range(2 ** d):
        for s in (0, 1):
            r = np.zeros(nvar); r[:m] = phi(y | (s << d), nv, M)
            if feas[y]:
                rows.append(r); lo.append(-np.inf); hi.append(0.0)                 # P <= 0
                bigm = B * r[:m].sum() + 1
                r2 = r.copy(); r2[m + 2 * fx.index(y) + s] = -bigm                  # P >= -M(1 - w)
                rows.append(r2); lo.append(-bigm); hi.append(np.inf)
            else:
                rows.append(r); lo.append(-np.inf); hi.append(-1.0)                # P <= -1
    for j in range(len(fx)):                                                       # one witness per satisfying y
        r = np.zeros(nvar); r[m + 2 * j: m + 2 * j + 2] = 1
        rows.append(r); lo.append(1.0); hi.append(np.inf)
    integ = np.r_[np.zeros(m), np.ones(2 * len(fx))]
    res = milp(np.zeros(nvar), constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integ,
               bounds=Bounds(np.r_[-B * np.ones(m), np.zeros(2 * len(fx))], np.r_[B * np.ones(m), np.ones(2 * len(fx))]),
               options={"time_limit": 300})
    return {0: True, 2: False}.get(res.status)     # None = undecided


if __name__ == "__main__":
    cache = json.load(open(PATH))
    out = {"entries": len(cache), "k_counts": {}, "problems": [], "k2_checks": []}
    for key, r in cache.items():
        d, pat = key.split(":"); d = int(d); feas = [c == "1" for c in pat]
        k = r["k"]
        out["k_counts"][str(k)] = out["k_counts"].get(str(k), 0) + 1
        if k == 0 and not k0_possible(d, feas):
            out["problems"].append([key, "cached k=0 but the LP says impossible"])
        if k >= 1 and k0_possible(d, feas):
            out["problems"].append([key, "k=0 would suffice"])
        if k == 2:
            c = {"pattern": key, "k1_possible_bound64": k1_possible(d, feas, 64.0),
                 "k1_possible_bound256": k1_possible(d, feas, 256.0)}
            out["k2_checks"].append(c)
            if c["k1_possible_bound64"] is not False or c["k1_possible_bound256"] is not False:
                out["problems"].append([key, "k=1 not ruled out", c])
    json.dump(out, open("results/iqp_minimality.json", "w"), indent=1)
    print("entries", out["entries"], "k counts", out["k_counts"], "problems", len(out["problems"]))
    print("k=1 entries: k=0 impossible (exact LP, unbounded coefficients)")
    print("k=2 entries: k=1 impossible at |coef| <= 64 and <= 256:",
          all(c["k1_possible_bound64"] is False and c["k1_possible_bound256"] is False for c in out["k2_checks"]))
