"""AML-derived settlement batches (public-data tier) and the extra classical baselines.

Source: IBM synthetic AML transactions, HI-Small (Altman et al., NeurIPS 2023 Datasets &
Benchmarks; Kaggle, CDLA-Sharing-1.0). Simulated customer payments, NOT real interbank or
RTGS data. Only USD -> USD Wire payments between two different banks are used.

Batch rule (direct mode, nothing fitted; parameters fixed in data/gen_aml_tier.py):
  epochs    = consecutive blocks of EPOCH_H hours
  groups    = in each epoch, disjoint groups of B banks built from payment COUNTS in that
              epoch (traffic_groups); amounts and solver results are not used
  batch     = consecutive payments (timestamp, then source row) between banks of one group
              inside one epoch; a batch is closed before the payment that would give some
              bank more than MAX_DEG payments or the batch more than N_MAX payments.
              Batches with fewer than N_MIN payments are discarded (counted).
  Every source row is in at most one batch.
Amounts: exact integer cents. `quantise` exists only for the quantisation study.
Liquidity: C_u = ceil(f * outgoing amount of u), same rule as the frozen benchmark.
"""
import math
from fractions import Fraction
from collections import Counter

import numpy as np

from settle.instances import Instance


# ---------- classical baselines (in addition to settle.exact.greedy) ----------

def _shortfalls(inst, x):
    net = inst.net_outflow_matrix() @ np.asarray(x, float)
    return net - np.asarray(inst.liquidity, float)


def _add_back(inst, x, order):
    """One pass in queue order: re-add any dropped payment that keeps x feasible."""
    x = x.copy()
    changed = True
    while changed:
        changed = False
        for k in order:
            if not x[k]:
                x[k] = 1
                if inst.is_feasible(x):
                    changed = True
                else:
                    x[k] = 0
    return x


def deletion_heuristic(inst):
    """Start from all payments. While some bank is short, take the bank with the largest
    shortfall s and drop its smallest selected outgoing payment with amount >= s
    (if none is that large, its largest one). Then add back in queue order."""
    x = np.ones(inst.n, int)
    while True:
        sf = _shortfalls(inst, x)
        u = int(np.argmax(sf))
        if sf[u] <= 1e-9:
            break
        outs = [k for k in inst.out_arcs(u) if x[k]]
        big = [k for k in outs if inst.arcs[k][2] >= sf[u] - 1e-9]
        k = min(big, key=lambda k: (inst.arcs[k][2], k)) if big else max(outs, key=lambda k: (inst.arcs[k][2], -k))
        x[k] = 0
    x = _add_back(inst, x, range(inst.n))
    assert inst.is_feasible(x)
    return inst.value(x), x


def bs_fifo(inst):
    """Bech & Soramaki (2001) style gridlock resolution with FIFO queues: try to settle the
    whole queue at once; while some banks are short, every short bank removes its most
    recently queued selected outgoing payment (arcs are stored in queue order). Then add
    back in queue order."""
    x = np.ones(inst.n, int)
    while True:
        sf = _shortfalls(inst, x)
        short = [u for u in range(inst.n_banks) if sf[u] > 1e-9]
        if not short:
            break
        for u in short:
            outs = [k for k in inst.out_arcs(u) if x[k]]
            x[max(outs)] = 0
    x = _add_back(inst, x, range(inst.n))
    assert inst.is_feasible(x)
    return inst.value(x), x


def exact_optimum(inst):
    """Like settle.exact.brute_force but ties by exact equality. Amounts here are integer
    cents (up to ~1e11), exactly representable in float64; np.isclose's default rtol=1e-5
    would count selections within ~1e-5 * opt of the optimum as optimal."""
    from settle.exact import feasibility_mask, values
    feas = feasibility_mask(inst)
    v = np.where(feas, values(inst), -np.inf)
    opt = v.max()
    return float(opt), [int(i) for i in np.flatnonzero(v == opt)]


# ---------- amounts and liquidity ----------

def quantise(raw, q):
    """Scale so max = q, round half up, floor at 1. Returns (ints, n_clamped)."""
    raw = np.asarray(raw, float)
    s = q / raw.max()
    v = np.floor(raw * s + 0.5).astype(int)
    clamped = int((v < 1).sum())
    return np.maximum(v, 1), clamped


def liquidity(n_banks, arcs, f):
    """C_u = ceil(f * outgoing of u), exact rational arithmetic (float f*o can land just
    above an integer, e.g. 0.3*10 = 3.0000000000000004)."""
    fr = Fraction(str(f))
    out = [0] * n_banks
    for a, b, w in arcs:
        out[a] += int(w)
    return [int(math.ceil(fr * o)) for o in out]


def lower_bound(n_banks, arcs):
    """Bech-Soramaki lower bound per bank: max(0, sent - received)."""
    net = np.zeros(n_banks)
    for a, b, w in arcs:
        net[a] += w
        net[b] -= w
    return [max(0.0, float(v)) for v in net]


# ---------- batches ----------

def traffic_groups(rows, b):
    """Disjoint groups of b banks built from payment COUNTS in this epoch only:
    seed with the unassigned pair with most payments between them, then repeatedly add
    the unassigned bank with most payments to/from the group; ties by bank id.
    Stops when no unassigned pair trades. Uses no amounts and no solver output."""
    pair = Counter()
    for r in rows:
        a, c = sorted((r["from_bank"], r["to_bank"]))
        pair[a, c] += 1
    free = {x for p in pair for x in p}
    groups = []
    while True:
        cand = [(n, p) for p, n in pair.items() if p[0] in free and p[1] in free]
        if not cand:
            break
        n, (a, c) = min(cand, key=lambda t: (-t[0], t[1]))
        G = [a, c]
        free -= {a, c}
        while len(G) < b:
            score = Counter()
            for (x, y), m in pair.items():
                if x in G and y in free:
                    score[y] += m
                elif y in G and x in free:
                    score[x] += m
            if not score:
                break
            nb = min(score, key=lambda z: (-score[z], z))
            G.append(nb)
            free.discard(nb)
        if len(G) == b:
            groups.append(G)
    return groups


def build_batches(rows, core, b, epoch_h, n_min, n_max, max_deg, seed, grouping="traffic"):
    """rows: list of dicts with keys row, minute (int minutes since start), from_bank,
    to_bank, amount; already restricted to the core and sorted by (minute, row).
    Returns (batches, counts)."""
    rng = np.random.default_rng(seed)
    core = list(core) if core is not None else None
    if grouping == "random" and core is None:
        raise ValueError("random grouping needs a core bank list")
    counts = Counter()
    batches = []
    by_epoch = {}
    for r in rows:
        by_epoch.setdefault(r["minute"] // (60 * epoch_h), []).append(r)
    n_epochs = max(by_epoch) + 1 if by_epoch else 0
    for ep in range(n_epochs):
        if grouping == "random":
            perm = [core[i] for i in rng.permutation(len(core))]   # one draw per epoch, always
            groups = [perm[i:i + b] for i in range(0, len(perm) - b + 1, b)]
        else:
            groups = traffic_groups(by_epoch.get(ep, []), b)
        gid = {bank: gi for gi, G in enumerate(groups) for bank in G}
        cur = {gi: [] for gi in range(len(groups))}
        deg = {gi: Counter() for gi in range(len(groups))}

        def close(gi):
            if len(cur[gi]) >= n_min:
                batches.append({"epoch": ep, "group": sorted(groups[gi]), "rows": list(cur[gi])})
                counts["batches"] += 1
            elif cur[gi]:
                counts["discarded_short"] += 1
                counts["rows_in_short"] += len(cur[gi])
            cur[gi] = []
            deg[gi] = Counter()

        for r in by_epoch.get(ep, []):
            ga, gb = gid.get(r["from_bank"]), gid.get(r["to_bank"])
            if ga is None or gb is None or ga != gb:
                counts["rows_cross_group"] += 1
                continue
            gi = ga
            d = deg[gi]
            if len(cur[gi]) >= n_max or d[r["from_bank"]] >= max_deg or d[r["to_bank"]] >= max_deg:
                close(gi)
                d = deg[gi]
            cur[gi].append(r)
            d[r["from_bank"]] += 1
            d[r["to_bank"]] += 1
        for gi in cur:
            close(gi)
    return batches, counts
