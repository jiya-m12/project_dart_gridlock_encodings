"""Main benchmark (offline). Protocol frozen in claude/build-log.md:
neighbour tuning (gamma=2), QAOA p=1..3 warm-started, COBYLA maxiter 1000 (amended from 300: at 300, 2-3 of 4 starts at p=3 stopped on the budget),
mixer start angles beta < 0 (amended in v7: earlier starts had beta > 0, the wrong quadrant for a minimised H;
see settle/qaoa.BETA_START_SIGN),
3 starts per p, objectives CVaR_0.25 (primary) and <H> (secondary), plus a
product-state (no entanglement) control with the same objective and budget.
E0 is simulated only where it has <= E0_MAX_QUBITS qubits.
Usage: python3 scripts/run_benchmark.py [--workers N] <set> [<set> ...]  (bench, t0, c, aml)
Default workers: all CPU cores. Results are deterministic per job (fixed seeds), independent of N.
Output: one JSON line per (instance, encoding, objective) in results/<set>_raw.jsonl."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import numpy as np
from settle.instances import load_instances_json
from settle.encodings import e0_standard, e1_iqp, e2_indicator, e3_unbalanced, e4_overflow_sq
from settle.verify import check_exact
from settle.qaoa import (Problem, optimise_sweep, statevector, logical_distribution,
                         optimise_product, product_probs)
from settle.metrics import quality
from settle import poly as P, iqp

E0_MAX_QUBITS = 18      # SPEC limit for exact simulation of the slack QUBO
P_MAX = 3
CACHE = iqp.load_cache_text(open("settle/iqp_data/iqp_cache.json").read())
E3W = {s: json.load(open("results/e3_tuning_%s.json" % s)) for s in ("bench", "t0", "c")}
OBJECTIVES = [("cvar", 0.25), ("mean", 1.0)]
# Protocol tag stored in every result row; run_benchmark.py refuses to append to a file with
# rows from another protocol, and analyse.py refuses to mix them.
PROTOCOL = "v7: beta starts < 0, COBYLA 1000/start"


def encodings(inst, set_name):
    """E3 only where weights were tuned for that set's amount units (not the cent-scale AML tier)."""
    d = {"E0": e0_standard(inst, rule="neighbour"),
         "E1": e1_iqp(inst, rule="neighbour", cache=dict(CACHE), derive_missing=False),
         "E2": e2_indicator(inst, rule="neighbour")}
    if set_name in E3W:
        d["E3"] = e3_unbalanced(inst, E3W[set_name]["l1"], E3W[set_name]["l2"])
    if set_name != "aml":
        # E4 at cent scale: normalised energy gaps ~1e-16 (below float64 resolution), so it is not run
        d["E4"] = e4_overflow_sq(inst, rule="neighbour")
    return d


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def dumps(rec):
    return json.dumps(rec, default=_jsonable)


def job(args):
    inst_d, enc_name, obj, alpha, set_name = args
    from settle.instances import Instance
    inst = Instance.from_dict(inst_d)
    enc = encodings(inst, set_name)[enc_name]
    t = time.time()
    z = P.to_ising(enc.poly)
    rec = {"protocol": PROTOCOL, "set": set_name, "instance": inst.name, "opt": inst.meta.get("opt"), "encoding": enc_name, "objective": obj, "alpha": alpha,
           "n_qubits": enc.n_total, "n_slack": enc.n_slack,
           "terms": P.term_stats(z), "exact": check_exact(inst, enc)["exact"]}
    zc = {k: v for k, v in z.items() if k}
    rec["coef_range"] = max(abs(v) for v in zc.values()) / min(abs(v) for v in zc.values())
    pr = Problem(enc)
    sweep = optimise_sweep(pr, P_MAX, objective=obj, alpha=alpha)
    rec["qaoa"] = {}
    for p, (E, g, b, ev, conv) in sweep.items():
        q = quality(inst, logical_distribution(pr, statevector(pr, g, b)))
        rec["qaoa"][p] = dict(q, objective_value=E, evals=ev, convergence=conv,
                              gammas=list(map(float, g)), betas=list(map(float, b)))
    Eps, th, evp, convp = optimise_product(pr, objective=obj, alpha=alpha)
    pp = product_probs(th)
    rec["product"] = dict(quality(inst, pp.reshape(-1, 2 ** pr.n_logical).sum(axis=0)),
                          objective_value=Eps, evals=evp, convergence=convp)
    rec["seconds"] = time.time() - t
    return rec


if __name__ == "__main__":
    from settle.instances import Instance
    args = sys.argv[1:]
    workers = os.cpu_count() or 2
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    sets = args or ["bench"]
    jobs = []
    for set_name in sets:
        out = "results/%s_raw.jsonl" % set_name
        done = set()
        if os.path.exists(out):
            for line in open(out):
                r = json.loads(line)
                if r.get("protocol") != PROTOCOL:
                    sys.exit("%s contains rows from a different protocol (%s). Move it to results/archive/ "
                             "before running, so old and new results are never mixed." % (out, r.get("protocol")))
                done.add((r["instance"], r["encoding"], r["objective"]))
        for inst in load_instances_json(open("data/%s_instances.json" % set_name).read()):
            for name, enc in encodings(inst, set_name).items():
                if name == "E0" and enc.n_total > E0_MAX_QUBITS:
                    continue
                for obj, a in OBJECTIVES:
                    if (inst.name, name, obj) not in done:
                        jobs.append((enc.n_total, (inst.to_dict(), name, obj, a, set_name)))
    jobs.sort(key=lambda j: -j[0])         # largest (slowest) jobs first: shortest total time on many cores
    print("jobs:", len(jobs), "workers:", workers, flush=True)
    import platform, scipy
    env = {"protocol": PROTOCOL, "sets": sets, "jobs": len(jobs), "workers": workers,
           "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "machine": platform.platform(), "cpu_count": os.cpu_count(),
           "started": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    json.dump(env, open("results/run_env_%s.json" % "_".join(sets), "w"), indent=1)
    files = {s: open("results/%s_raw.jsonl" % s, "a") for s in sets}
    with Pool(workers) as pool:
        for rec in pool.imap_unordered(job, [j[1] for j in jobs]):
            f = files[rec["set"]]
            f.write(dumps(rec) + "\n"); f.flush()
            print(rec["set"], rec["instance"], rec["encoding"], rec["objective"], rec["n_qubits"], "q",
                  "P(opt) p3 %.3f prod %.3f" % (rec["qaoa"][P_MAX]["p_opt"], rec["product"]["p_opt"]),
                  "%.0fs" % rec["seconds"], flush=True)
