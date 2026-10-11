"""Build the AML-derived (public-data) tier. Offline, run from the repo root:

    python3 data/gen_aml_tier.py path/to/aml_usd_interbank.csv.gz

Input: the filtered file written by data/aml_profile_filter.py from HI-Small_Trans.csv
(IBM synthetic AML transactions, Altman et al., NeurIPS 2023 D&B, CDLA-Sharing-1.0).
Outputs (amounts in integer US cents, exactly as in the source rows):
  data/aml_pool.json            every batch, at its seeded liquidity level
  data/aml_instances.json       the "hard" subset used for QAOA: opt > value-ordered greedy
                                (same filter as data/bench_instances.json)
  results/aml_tier_summary.json counts, liquidity sweep, quantisation study, statistics

All rule parameters are fixed below. They were chosen after looking at batch COUNTS only
(no solver results on the QAOA side, no QAOA runs): see DATA.md.
"""
import csv, datetime as dt, gzip, hashlib, json, math, os, sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settle.instances import Instance                                    # noqa: E402
from settle.exact import greedy, feasibility_mask, values    # noqa: E402
from settle.aml import (build_batches, deletion_heuristic, bs_fifo, liquidity, exact_optimum,  # noqa: E402
                        lower_bound, quantise)

# ---- frozen rule parameters ----
FORMAT = "Wire"
GROUP_SIZE = 5
EPOCH_HOURS = 12
N_MIN, N_MAX, MAX_DEG = 6, 9, 5
F_GRID = (0.2, 0.3, 0.4)          # same grid as the frozen benchmark
F_SWEEP = (0.1, 0.2, 0.3, 0.4)    # descriptive sweep only
Q_STUDY = (10, 100, 1000, 10 ** 4, 10 ** 6)
SEED = 20261010
T0 = dt.datetime(2022, 9, 1)
SOURCE_CSV_SHA256 = "b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040"  # HI-Small_Trans.csv
LICENCE = ("Derived from IBM 'Transactions for Anti Money Laundering' HI-Small (Altman et al., "
           "NeurIPS 2023 Datasets and Benchmarks; Kaggle). Licensed under CDLA-Sharing-1.0 "
           "(https://cdla.dev/sharing-1-0/). MODIFIED: rows filtered (USD Wire, inter-bank), "
           "grouped into settlement batches, banks renumbered, amounts in cents, liquidity added "
           "by a stated rule. Synthetic data; not real or RTGS payments.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cents(s):
    """'1234.56' -> 123456 exactly (no float rounding)."""
    s = s.strip()
    if "." in s:
        a, b = s.split(".")
        b = (b + "00")[:2] if len(b) <= 2 else None
        if b is None:
            raise ValueError("more than 2 decimals: " + s)
        return int(a or "0") * 100 + int(b)
    return int(s) * 100


def load_rows(path):
    rows, n_all = [], 0
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            n_all += 1
            if r["format"] != FORMAT:
                continue
            assert r["from_bank"] != r["to_bank"]
            assert r["amount_paid"] == r["amount_received"]
            t = dt.datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M")
            rows.append({"row": int(r["row"]), "minute": int((t - T0).total_seconds() // 60),
                         "from_bank": r["from_bank"], "to_bank": r["to_bank"],
                         "cents": cents(r["amount_paid"]), "is_laundering": int(r["is_laundering"])})
    rows.sort(key=lambda r: (r["minute"], r["row"]))
    return rows, n_all


def make_instance(batch, name, f):
    ids = {}
    for r in batch["rows"]:
        for bk in (r["from_bank"], r["to_bank"]):
            ids.setdefault(bk, len(ids))
    arcs = [(ids[r["from_bank"]], ids[r["to_bank"]], r["cents"]) for r in batch["rows"]]
    t0 = batch["rows"][0]["minute"]
    meta = {"source": "IBM AML HI-Small (synthetic), USD Wire inter-bank", "unit": "US cents",
            "licence": "CDLA-Sharing-1.0, modified; see data/AML_DATA_LICENSE.txt",
            "source_rows": [r["row"] for r in batch["rows"]],
            "epoch": batch["epoch"], "start": str(T0 + dt.timedelta(minutes=t0)),
            "span_minutes": batch["rows"][-1]["minute"] - t0, "f_liq": f,
            "queue_order": "arcs are in submission order (timestamp, then source row)"}
    return Instance(name, len(ids), arcs, liquidity(len(ids), arcs, f), meta)


def classify(inst):
    """Exact optimum and all baselines. Brute force (n <= 9)."""
    lb = lower_bound(inst.n_banks, inst.arcs)
    trivial = all(inst.liquidity[u] >= lb[u] for u in range(inst.n_banks))
    opt, idx = exact_optimum(inst)
    r = {"opt": opt, "n_opt_solutions": len(idx), "trivial_all_settle": trivial,
         "greedy": greedy(inst)[0], "greedy_fifo": greedy(inst, "fifo")[0],
         "deletion": deletion_heuristic(inst)[0], "bs_fifo": bs_fifo(inst)[0],
         "total": float(sum(w for _, _, w in inst.arcs)),
         "lower_bound": lb}
    tol = 1e-9
    r["hard"] = (not trivial) and opt > r["greedy"] + tol
    r["beats_fifo"] = opt > r["greedy_fifo"] + tol
    r["beats_deletion"] = opt > r["deletion"] + tol
    r["beats_bs_fifo"] = opt > r["bs_fifo"] + tol
    r["beats_all_heuristics"] = opt > max(r["greedy"], r["greedy_fifo"], r["deletion"], r["bs_fifo"]) + tol
    return r


def has_directed_cycle(inst):
    adj = {u: set() for u in range(inst.n_banks)}
    for a, b, _ in inst.arcs:
        adj[a].add(b)
    color = {}

    def dfs(u):
        color[u] = 1
        for v in adj[u]:
            if color.get(v) == 1 or (v not in color and dfs(v)):
                return True
        color[u] = 2
        return False
    return any(u not in color and dfs(u) for u in adj)


def quant_preserved(inst, q):
    """Is some optimal set of the q-quantised instance optimal for the exact (cents) one?"""
    wq, _ = quantise([w for _, _, w in inst.arcs], q)
    arcs_q = [(a, b, int(w)) for (a, b, _), w in zip(inst.arcs, wq)]
    iq = Instance("q", inst.n_banks, arcs_q, liquidity(inst.n_banks, arcs_q, inst.meta["f_liq"]))
    opt_q, idx = exact_optimum(iq)
    fe, va = feasibility_mask(inst), values(inst)
    opt = va[fe].max()
    hard_q = opt_q > greedy(iq)[0]
    return any(fe[j] and va[j] == opt for j in idx), hard_q


def main(path):
    rows, n_all = load_rows(path)
    assert all(r["is_laundering"] == 0 for r in rows), "laundering rows in Wire subset"
    batches, counts = build_batches(rows, None, GROUP_SIZE, EPOCH_HOURS, N_MIN, N_MAX, MAX_DEG,
                                    SEED, grouping="traffic")
    rng = np.random.default_rng(SEED)
    pool, hard = [], []
    sweep = {f: Counter() for f in F_SWEEP}
    qstudy = {q: Counter() for q in Q_STUDY}
    stats = Counter()
    ratios, recips = [], []
    for i, B in enumerate(batches):
        f = float(F_GRID[rng.integers(len(F_GRID))])   # one draw per batch, always
        inst = make_instance(B, "AML%03d" % i, f)
        c = classify(inst)
        inst.meta.update({k: v for k, v in c.items()})
        pool.append(inst)
        if c["hard"]:
            hard.append(inst)
        # descriptive statistics (do not select anything)
        ws = [w for _, _, w in inst.arcs]
        ratios.append(max(ws) / min(ws))
        pairs = {(a, b) for a, b, _ in inst.arcs}
        recips.append(sum((b, a) in pairs for a, b in pairs) / len(pairs))
        stats["has_directed_cycle"] += has_directed_cycle(inst)
        stats["n_%d" % inst.n] += 1
        stats["banks_%d" % inst.n_banks] += 1
        for fs in F_SWEEP:
            j = Instance("s", inst.n_banks, inst.arcs, liquidity(inst.n_banks, inst.arcs, fs), {"f_liq": fs})
            cs = classify(j)
            sweep[fs]["batches"] += 1
            for k in ("trivial_all_settle", "hard", "beats_fifo", "beats_deletion", "beats_bs_fifo",
                      "beats_all_heuristics"):
                sweep[fs][k] += bool(cs[k])
        for q in Q_STUDY:
            pres, hq = quant_preserved(inst, q)
            qstudy[q]["optimum_preserved"] += pres
            qstudy[q]["hard_after_quantising"] += hq
            if c["hard"]:
                qstudy[q]["optimum_preserved_on_hard"] += pres

    allw = [r["cents"] for r in rows]
    summary = {
        "licence": LICENCE,
        "provenance": {"source_csv_sha256": SOURCE_CSV_SHA256, "filtered_gz_sha256": sha256(path),
                       "generator_sha256": sha256(os.path.abspath(__file__)),
                       "settle_aml_sha256": sha256(os.path.join(os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))), "settle", "aml.py"))},
        "rule": {"format": FORMAT, "currency": "US Dollar (both sides)", "inter_bank_only": True,
                 "grouping": "traffic", "group_size": GROUP_SIZE, "epoch_hours": EPOCH_HOURS,
                 "n_min": N_MIN, "n_max": N_MAX, "max_degree": MAX_DEG, "f_grid": F_GRID,
                 "liquidity": "C_u = ceil(f * outgoing cents of u)", "seed": SEED,
                 "hard_filter": "not trivially all-settle and opt > value-ordered greedy"},
        "counts": {"filtered_rows_in_gz": n_all, "wire_rows": len(rows),
                   "rows_in_batches": sum(i.n for i in pool), **dict(counts),
                   "pool": len(pool), "hard": len(hard),
                   "hard_beats_deletion": sum(i.meta["beats_deletion"] for i in hard),
                   "hard_beats_bs_fifo": sum(i.meta["beats_bs_fifo"] for i in hard),
                   "hard_beats_all_heuristics": sum(i.meta["beats_all_heuristics"] for i in hard),
                   "pool_f": dict(Counter(i.meta["f_liq"] for i in pool)),
                   "hard_f": dict(Counter(i.meta["f_liq"] for i in hard))},
        "liquidity_sweep_all_batches": {str(f): dict(v) for f, v in sweep.items()},
        "quantisation_study_vs_exact_cents": {str(q): dict(v) for q, v in qstudy.items()},
        "statistics": {
            "wire_amount_dollars_mean": float(np.mean(allw)) / 100,
            "wire_amount_dollars_median": float(np.median(allw)) / 100,
            "mean_over_median": float(np.mean(allw) / np.median(allw)),
            "batch_max_over_min_quantiles": {str(q): float(np.quantile(ratios, q)) for q in (.1, .25, .5, .75, .9)},
            "batch_reciprocity_mean": float(np.mean(recips)),
            **dict(stats)},
    }
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name, insts in (("aml_pool.json", pool), ("aml_instances.json", hard)):
        with open(os.path.join(root, "data", name), "w") as f:   # same list schema as bench
            json.dump([i.to_dict() for i in insts], f, separators=(",", ":"))
    with open(os.path.join(root, "data", "AML_DATA_LICENSE.txt"), "w") as f:
        f.write(LICENCE + "\nApplies to data/aml_pool.json and data/aml_instances.json. "
                "Code in this repository is MIT.\n")
    with open(os.path.join(root, "results", "aml_tier_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["counts"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
