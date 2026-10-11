"""Summarise results/<set>_raw.jsonl for set in (bench, t0, c) ->
results/summary.json (machine-readable) and results/bench_summary.json (text lines
for main.py mode=results). Medians with 95% bootstrap CIs over instances
(2000 resamples, seed 0); paired Wilcoxon signed-rank tests between encodings."""
import os, sys, json
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

SETS = {"bench": "main set: 40 instances, 4-5 banks, 6-8 payments, amounts 1-10",
        "t0": "added tier: 20 instances, 4 banks, 6 payments, amounts 1-3 (E0 fits in <=18 qubits)",
        "c": "added tier: 20 instances, 6 banks, 10 payments, amounts 1-10",
        "aml": "public-data tier: 22 hard batches of USD wire payments from the IBM AML dataset, exact cents (DATA.md)"}
ENCS = ["E0", "E1", "E2", "E3", "E4"]
rng = np.random.default_rng(0)


def med_ci(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return [float("nan")] * 3
    bs = np.median(rng.choice(v, size=(2000, len(v)), replace=True), axis=1)
    return [float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def load(set_name):
    path = "results/%s_raw.jsonl" % set_name
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path)]
    protos = {r.get("protocol") for r in rows}
    if len(protos) > 1:
        sys.exit("%s mixes protocols %s; refusing to summarise." % (path, protos))
    keys = [(r["instance"], r["encoding"], r["objective"]) for r in rows]
    if len(keys) != len(set(keys)):
        sys.exit("%s has duplicate (instance, encoding, objective) rows." % path)
    return rows


def summarise(rows):
    out = {}
    g = defaultdict(list)
    for r in rows:
        g[(r["encoding"], r["objective"])].append(r)
    for (enc, obj), R in sorted(g.items()):
        d = {"n": len(R), "exact": sum(r["exact"] for r in R),
             "qubits_median": float(np.median([r["n_qubits"] for r in R])),
             "cnot_median": float(np.median([r["terms"]["cnot_ladder"] for r in R])),
             "coef_range_median": float(np.median([r["coef_range"] for r in R])), "p": {}}
        for p in ("1", "2", "3"):
            d["p"][p] = {"p_opt": med_ci([r["qaoa"][p]["p_opt"] for r in R]),
                         "enrichment": med_ci([r["qaoa"][p]["enrichment"] for r in R]),
                         "p_feasible": med_ci([r["qaoa"][p]["p_feasible"] for r in R]),
                         "exp_ratio": med_ci([r["qaoa"][p]["exp_ratio"] for r in R]),
                         "starts_hit_maxiter": int(sum(r["qaoa"][p]["convergence"]["hit_maxiter"] for r in R))}
        d["product"] = {"p_opt": med_ci([r["product"]["p_opt"] for r in R]),
                        "p_opt_gt_0.9": int(sum(r["product"]["p_opt"] > 0.9 for r in R)),
                        "beats_qaoa_p3": int(sum(r["product"]["p_opt"] > r["qaoa"]["3"]["p_opt"] for r in R))}
        out["%s|%s" % (enc, obj)] = d
    return out


def paired(rows, obj="cvar", p="3", metric="p_opt"):
    D = {(r["instance"], r["encoding"]): r for r in rows if r["objective"] == obj}
    insts = sorted({r["instance"] for r in rows})
    res = {}
    for a in ENCS:
        for b in ENCS:
            if a >= b:
                continue
            common = [i for i in insts if (i, a) in D and (i, b) in D]
            if len(common) < 5:
                continue
            x = np.array([D[(i, a)]["qaoa"][p][metric] for i in common])
            y = np.array([D[(i, b)]["qaoa"][p][metric] for i in common])
            nz = np.abs(x - y) > 1e-9
            pv = float(wilcoxon(x[nz], y[nz]).pvalue) if nz.sum() >= 5 else float("nan")
            res["%s_vs_%s" % (a, b)] = {"n": len(common), "a_better": int((x - y > 1e-9).sum()),
                                        "b_better": int((y - x > 1e-9).sum()), "wilcoxon_p": pv}
    # Holm correction over all pairs tested in this set (controls the family-wise error rate)
    ks = sorted((k for k in res if res[k]["wilcoxon_p"] == res[k]["wilcoxon_p"]), key=lambda k: res[k]["wilcoxon_p"])
    run = 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (len(ks) - i) * res[k]["wilcoxon_p"]))
        res[k]["holm_p"] = run
    return res


def best_vs_best(rows):
    """Per instance: best product-state P(opt) over (encoding, objective) vs best QAOA p=3
    P(opt) over (encoding, objective). Also per encoding (both objectives pooled)."""
    insts = sorted({r["instance"] for r in rows})
    out = {"n": len(insts), "product_better": 0, "qaoa_better": 0, "per_encoding": {}}
    for i in insts:
        R = [r for r in rows if r["instance"] == i]
        bp = max(r["product"]["p_opt"] for r in R)
        bq = max(r["qaoa"]["3"]["p_opt"] for r in R)
        out["product_better" if bp > bq else "qaoa_better"] += 1
    for e in ENCS:
        R = [r for r in rows if r["encoding"] == e]
        if not R:
            continue
        ii = sorted({r["instance"] for r in R})
        pb = sum(max(r["product"]["p_opt"] for r in R if r["instance"] == i) >
                 max(r["qaoa"]["3"]["p_opt"] for r in R if r["instance"] == i) for i in ii)
        out["per_encoding"][e] = {"n": len(ii), "product_better": int(pb)}
    return out


def visibility(set_name):
    """How visible the settlement value is to QAOA after the normalisation (H / max|coef|):
    max |objective Ising coefficient| / max |total Ising coefficient|, median over instances.
    Encodings whose penalties are much larger than the objective (E0, E4) leave the objective
    small after normalisation; this is a scale effect separate from the penalty's shape."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from settle.instances import load_instances_json
    from settle.encodings import e0_standard, e1_iqp, e2_indicator, e3_unbalanced, e4_overflow_sq
    from settle import iqp, poly as Pp
    cache = iqp.load_cache_text(open("settle/iqp_data/iqp_cache.json").read())
    e3p = "results/e3_tuning_%s.json" % set_name
    e3 = json.load(open(e3p)) if os.path.exists(e3p) else None
    I = load_instances_json(open("data/%s_instances.json" % set_name).read())
    mk = {"E0": lambda i: e0_standard(i, rule="neighbour"),
          "E1": lambda i: e1_iqp(i, rule="neighbour", cache=dict(cache), derive_missing=False),
          "E2": lambda i: e2_indicator(i, rule="neighbour"),
          "E3": (lambda i: e3_unbalanced(i, e3["l1"], e3["l2"])) if e3 else None,
          "E4": lambda i: e4_overflow_sq(i, rule="neighbour")}
    out = {}
    for e, f in mk.items():
        if f is None:
            continue
        v = []
        for i in I:
            enc = f(i)
            zo = Pp.to_ising(enc.objective); zo.pop((), None)
            zt = Pp.to_ising(enc.poly); zt.pop((), None)
            v.append(max(abs(c) for c in zo.values()) / max(abs(c) for c in zt.values()))
        out[e] = float(np.median(v))
    return out


def near_alpha(rows, alpha=0.25, frac=0.95):
    """CVaR_alpha ceiling. CVaR_alpha is the mean cost of the best alpha fraction of samples, so once
    the optimum holds ~alpha of the probability the objective cannot improve and P(opt) is no
    longer rewarded. Count CVaR runs (p = 3) whose P(opt) reached frac * alpha."""
    out = {}
    for e in ENCS:
        R = [r for r in rows if r["encoding"] == e and r["objective"] == "cvar"]
        if R:
            out[e] = {"n": len(R), "reached": int(sum(r["qaoa"]["3"]["p_opt"] >= frac * alpha for r in R)),
                      "threshold": frac * alpha}
    return out


def text_lines(set_name, S, P):
    L = ["", "== %s (%s) ==" % (set_name, SETS[set_name]),
         "enc obj   exact  qubits CNOT/layer | P(opt) at p=1, 2, 3 [95% CI at p=3] | x random (p=3) | P(feas) p=3 | product state P(opt)"]
    for key, d in S.items():
        enc, obj = key.split("|")
        p3 = d["p"]["3"]
        L.append("%-3s %-5s %2d/%-2d %4.0f %6.0f      | %.3f %.3f %.3f [%.3f,%.3f] | %6.1fx | %.3f | %.3f (>0.9 on %d/%d)" % (
            enc, obj, d["exact"], d["n"], d["qubits_median"], d["cnot_median"],
            d["p"]["1"]["p_opt"][0], d["p"]["2"]["p_opt"][0], p3["p_opt"][0], p3["p_opt"][1], p3["p_opt"][2],
            p3["enrichment"][0], p3["p_feasible"][0], d["product"]["p_opt"][0], d["product"]["p_opt_gt_0.9"], d["n"]))
    L.append("paired, CVaR, p=3, P(opt): " + "; ".join(
        "%s %d-%d (p=%.1g, Holm %.1g)" % (k.replace("_vs_", " vs "), v["a_better"], v["b_better"], v["wilcoxon_p"], v.get("holm_p", float("nan"))) for k, v in P.items()))
    return L


if __name__ == "__main__":
    summary, lines = {}, ["Medians over instances. 'x random' = P(opt) / P(opt) of uniform random guessing.",
                          "QAOA: X mixer, p=1..3 warm-started, COBYLA (1000 iter/start), neighbour penalty weights.",
                          "Product state = one RY per qubit, no entanglement, same objective and optimiser (control)."]
    for s in SETS:
        rows = load(s)
        if not rows:
            continue
        S = summarise(rows)
        P = paired(rows)
        B = best_vs_best(rows)
        F = near_alpha(rows)
        V = visibility(s)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from settle.instances import load_instances_json
        from settle.encodings import e0_standard
        insts = load_instances_json(open("data/%s_instances.json" % s).read())
        e0_eligible = sum(e0_standard(i, rule="neighbour").n_total <= 18 for i in insts)
        summary[s] = {"description": SETS[s], "by_encoding": S, "paired_cvar_p3": P,
                      "paired_mean_p3": paired(rows, obj="mean"),
                      "best_product_vs_best_qaoa": B, "p_opt_near_alpha_p3": F, "objective_visibility_median": V,
                      "e0_eligible_le18_qubits": int(e0_eligible), "n_instances": len(insts)}
        lines += text_lines(s, S, P)
        lines.append("best product state vs best QAOA (any encoding/objective): product better on %d/%d instances"
                     % (B["product_better"], B["n"]))
        lines.append("CVaR runs with P(opt) >= 0.95*alpha at p=3 (the CVaR ceiling): " +
                     ", ".join("%s %d/%d" % (e, v["reached"], v["n"]) for e, v in F.items()))
    json.dump(summary, open("results/summary.json", "w"), indent=1)
    json.dump({"header": "Offline benchmark (scripts/run_benchmark.py, scripts/analyse.py)", "lines": lines},
              open("results/bench_summary.json", "w"), indent=0)
    print("\n".join(lines))
