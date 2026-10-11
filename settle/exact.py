"""Ground truth and classical baselines."""
import numpy as np


def all_bitstrings(n):
    """(2^n, n) array; row i is the binary expansion of i, bit j = (i >> j) & 1
    (qubit j = least significant first, the Qiskit convention)."""
    idx = np.arange(2 ** n, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)


def feasibility_mask(inst):
    """Boolean vector over all 2^n payment selections: feasible or not."""
    X = all_bitstrings(inst.n).astype(float)
    net = X @ inst.net_outflow_matrix().T
    return np.all(net <= np.asarray(inst.liquidity, float) + 1e-9, axis=1)


def values(inst):
    return all_bitstrings(inst.n).astype(float) @ inst.weights


# Two settlement values count as equal only if they differ by less than OPT_ATOL (absolute).
# Amounts are integers (controlled tiers: 1..10; AML tier: integer cents), so distinct values
# differ by >= 1. np.isclose's default relative tolerance (1e-5 * opt) would wrongly merge
# near-optimal states with the optimum at cent scale (opt ~ 1e5 cents -> tolerance ~1 cent).
OPT_ATOL = 1e-6


def is_value(v, target):
    """Elementwise |v - target| <= OPT_ATOL (no relative tolerance)."""
    return np.abs(np.asarray(v, float) - float(target)) <= OPT_ATOL


def brute_force(inst):
    """Exact optimum by enumeration (n <= ~22). Returns (opt_value, list of optimal int indices)."""
    feas = feasibility_mask(inst)
    v = np.where(feas, values(inst), -np.inf)
    opt = v.max()
    return float(opt), [int(i) for i in np.flatnonzero(is_value(v, opt))]


def milp(inst):
    """Exact optimum with scipy's MILP (HiGHS). Returns (opt_value, x)."""
    from scipy.optimize import milp as _milp, LinearConstraint, Bounds
    A = inst.net_outflow_matrix()
    r = _milp(c=-inst.weights, constraints=LinearConstraint(A, -np.inf, np.asarray(inst.liquidity, float)),
              integrality=np.ones(inst.n), bounds=Bounds(0, 1), options={"mip_rel_gap": 0.0})   # prove optimality
    if r.status != 0:
        raise RuntimeError("MILP failed: " + r.message)
    x = np.round(r.x).astype(int)
    return inst.value(x), x


def greedy(inst, order="value"):
    """RTGS-style queue processing: go through the queue (largest first, or FIFO),
    settle a payment if the payer's current balance covers it; repeat passes until
    nothing more settles. Receives update balances. Cannot resolve gridlock cycles
    where every bank needs an incoming payment first."""
    bal = np.array(inst.liquidity, float)
    x = np.zeros(inst.n, int)
    ks = list(range(inst.n))
    if order == "value":
        ks.sort(key=lambda k: -inst.arcs[k][2])
    changed = True
    while changed:
        changed = False
        for k in ks:
            if x[k]:
                continue
            a, b, w = inst.arcs[k]
            if bal[a] >= w:
                bal[a] -= w
                bal[b] += w
                x[k] = 1
                changed = True
    assert inst.is_feasible(x)
    return inst.value(x), x


def deletion_heuristic(inst):
    """Simple multilateral heuristic (stronger than the RTGS queue): start from
    settling everything; while some bank is short, take the bank with the largest
    shortfall and drop the smallest of its settled outgoing payments that covers
    the shortfall (or its largest one if none covers it). Then try to add dropped
    payments back, largest first, keeping feasibility."""
    A = inst.net_outflow_matrix()
    C = np.asarray(inst.liquidity, float)
    x = np.ones(inst.n, int)
    while True:
        short = A @ x - C
        u = int(np.argmax(short))
        if short[u] <= 1e-9:
            break
        outs = [k for k in inst.out_arcs(u) if x[k]]
        cover = [k for k in outs if inst.arcs[k][2] >= short[u]]
        k = min(cover, key=lambda k: inst.arcs[k][2]) if cover else max(outs, key=lambda k: inst.arcs[k][2])
        x[k] = 0
    for k in sorted(np.flatnonzero(x == 0), key=lambda k: -inst.arcs[k][2]):
        x[k] = 1
        if not inst.is_feasible(x):
            x[k] = 0
    assert inst.is_feasible(x)
    return inst.value(x), x
