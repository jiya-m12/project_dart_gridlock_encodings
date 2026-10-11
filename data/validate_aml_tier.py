"""Independent validation of the AML tier. Does NOT import settle.aml or settle.exact for
the core checks: instances are rebuilt from the source rows and solved by its own
integer enumeration.

    python3 data/validate_aml_tier.py path/to/aml_usd_interbank.csv.gz
"""
import csv, datetime as dt, gzip, itertools, json, math, os, sys
from collections import Counter
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = dt.datetime(2022, 9, 1)
fails = Counter()


def check(cond, what):
    if not cond:
        fails[what] += 1


def to_cents(s):
    d = Fraction(s)
    assert (d * 100).denominator == 1, s
    return int(d * 100)


def opt_int(nb, arcs, C):
    """Exact optimum over all 2^n selections, integer arithmetic. Returns (opt, set of optimal masks)."""
    n = len(arcs)
    best, arg = -1, []
    for m in range(2 ** n):
        net = [0] * nb
        val = 0
        for k, (a, b, w) in enumerate(arcs):
            if (m >> k) & 1:
                net[a] += w
                net[b] -= w
                val += w
        if all(net[u] <= C[u] for u in range(nb)):
            if val > best:
                best, arg = val, [m]
            elif val == best:
                arg.append(m)
    return best, arg


def feasible_int(nb, arcs, C, x):
    net = [0] * nb
    for k, (a, b, w) in enumerate(arcs):
        if x[k]:
            net[a] += w
            net[b] -= w
    return all(net[u] <= C[u] for u in range(nb))


def main(gz):
    src = {}
    with gzip.open(gz, "rt", newline="") as f:
        for r in csv.DictReader(f):
            src[int(r["row"])] = r
    pool = json.load(open(os.path.join(ROOT, "data", "aml_pool.json")))
    hard = json.load(open(os.path.join(ROOT, "data", "aml_instances.json")))
    summ = json.load(open(os.path.join(ROOT, "results", "aml_tier_summary.json")))
    used = Counter()
    n_hard_recount = 0
    for d in pool:
        m = d["meta"]
        rows = [src[i] for i in m["source_rows"]]
        used.update(m["source_rows"])
        # source filter
        for r in rows:
            check(r["format"] == "Wire", "format")
            check(r["from_bank"] != r["to_bank"], "intra-bank row")
            check(r["is_laundering"] == "0", "laundering row")
            check(r["amount_paid"] == r["amount_received"], "paid != received")
        # queue order and single epoch
        mins = [int((dt.datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M") - T0).total_seconds() // 60)
                for r in rows]
        keys = list(zip(mins, m["source_rows"]))
        check(keys == sorted(keys), "queue order")
        check(len({x // 720 for x in mins}) == 1 and mins[0] // 720 == m["epoch"], "epoch")
        check(m["span_minutes"] == mins[-1] - mins[0], "span")
        # rebuild arcs: banks renumbered by first appearance
        ids = {}
        for r in rows:
            for bk in (r["from_bank"], r["to_bank"]):
                ids.setdefault(bk, len(ids))
        arcs = [(ids[r["from_bank"]], ids[r["to_bank"]], to_cents(r["amount_paid"])) for r in rows]
        check([tuple(a) for a in d["arcs"]] == arcs, "arcs differ from source")
        check(d["n_banks"] == len(ids) <= 5, "bank count")
        check(6 <= len(arcs) <= 9, "batch size")
        deg = Counter()
        for a, b, _ in arcs:
            deg[a] += 1
            deg[b] += 1
        check(max(deg.values()) <= 5, "degree cap")
        # liquidity, exact rational
        f = Fraction(str(m["f_liq"]))
        check(m["f_liq"] in (0.2, 0.3, 0.4), "f grid")
        out = [0] * len(ids)
        net_all = [0] * len(ids)
        for a, b, w in arcs:
            out[a] += w
            net_all[a] += w
            net_all[b] -= w
        C = [math.ceil(f * o) for o in out]
        check(C == d["liquidity"], "liquidity rule")
        trivial = all(C[u] >= max(0, net_all[u]) for u in range(len(ids)))
        check(trivial == m["trivial_all_settle"], "trivial flag")
        # exact optimum, independent
        opt, argm = opt_int(len(ids), arcs, C)
        check(opt == m["opt"], "opt")
        check(len(argm) == m["n_opt_solutions"], "n optimal solutions")
        for key in ("greedy", "greedy_fifo", "deletion", "bs_fifo"):
            check(m[key] <= opt, "baseline above opt: " + key)
        hard_flag = (not trivial) and opt > m["greedy"]
        check(hard_flag == m["hard"], "hard flag")
        n_hard_recount += hard_flag
    check(max(used.values()) == 1, "source row reused")
    check(n_hard_recount == len(hard) == summ["counts"]["hard"], "hard count")
    hard_names = {d["name"] for d in hard}
    check(hard_names == {d["name"] for d in pool if d["meta"]["hard"]}, "hard subset")
    for p in ("aml_pool.json", "aml_instances.json"):
        check(os.path.getsize(os.path.join(ROOT, "data", p)) < 2_000_000, "file size")
    print("instances checked:", len(pool), "hard:", len(hard), "distinct source rows:", len(used))
    print("FAILURES:", dict(fails) if fails else "none")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
