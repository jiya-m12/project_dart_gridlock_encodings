"""Write the results tables into RESULTS.md from results/summary.json
(between <!-- RESULTS --> markers), so the documents never drift from the data."""
import json, re

S = json.load(open("results/summary.json"))
NAMES = {"E0": "E0 standard slack", "E1": "E1 IQP", "E2": "E2 indicator", "E3": "E3 unbalanced", "E4": "E4 overflow²"}
TITLE = {"bench": "Main set (40 instances, 4-5 banks, 6-8 payments, amounts 1-10)",
         "t0": "Tier T0 (20 instances, 4 banks, 6 payments, amounts 1-3; E0 fits in ≤ 18 qubits)",
         "c": "Tier C (20 instances, 6 banks, 10 payments, amounts 1-10)",
         "aml": "Public-data tier AML (22 hard batches, 4-5 banks, 6-9 payments, exact US cents; IBM AML dataset, see DATA.md)"}


def table(s):
    d = S[s]["by_encoding"]
    L = ["**%s**" % TITLE[s], "",
         "| Encoding | exact | qubits | CNOT/layer | P(opt) p=3, CVaR [95% CI] | × random | P(feasible) | product state P(opt) (⟨H⟩) |",
         "|---|---|---|---|---|---|---|---|"]
    elig = S[s].get("e0_eligible_le18_qubits")
    for e in ["E0", "E1", "E2", "E3", "E4"]:
        k = "%s|cvar" % e
        if k not in d:
            if e == "E3" and s == "aml":
                why = "n/a: E3 weights were tuned in small-integer units and do not transfer to cents"
            elif e == "E4" and s == "aml":
                why = "n/a: at cents the optimum and other settlements differ by ~1e-16 after normalisation (float64 limit)"
            else:
                why = "not simulated (> 18 qubits on all %d instances)" % S[s].get("n_instances", 0)
            L.append("| %s | – | – | – | %s | | | |" % (NAMES[e], why)); continue
        if e == "E0" and elig is not None and d[k]["n"] < elig:
            L.append("| %s | – | – | – | only %d of the %d instances that fit in 18 qubits have finished; "
                     "results pending | | | |" % (NAMES[e], d[k]["n"], elig)); continue
        x, m = d[k], d["%s|mean" % e]
        po = x["p"]["3"]["p_opt"]
        ci = "[%.3f, %.3f]" % (po[1], po[2]) if x["n"] >= 5 else "(n = %d, no CI)" % x["n"]
        L.append("| %s | %d/%d | %g | %g | %.3f %s | %.1f× | %.2f | %.2f (> 0.9 on %d/%d) |" % (
            NAMES[e], x["exact"], x["n"], x["qubits_median"], x["cnot_median"], po[0], ci,
            x["p"]["3"]["enrichment"][0], x["p"]["3"]["p_feasible"][0], m["product"]["p_opt"][0],
            m["product"]["p_opt_gt_0.9"], m["n"]))
    P = S[s]["paired_cvar_p3"]
    pairs = ["%s better on %d/%d (Wilcoxon p = %.1g)" % (k.replace("_vs_", " vs ").replace("E2 vs ", "E2 vs ") , v["a_better"] if k.startswith("E2") else v["b_better"], v["n"], v["wilcoxon_p"])
             for k, v in P.items() if "E2" in k]
    L += ["", "Paired, P(opt) at p = 3, CVaR — E2 compared with each other encoding: " +
          "; ".join(("E2 vs %s: E2 better on %d/%d (Wilcoxon p = %.1g, Holm-adjusted %.1g)" % (k.replace("E2_vs_", "").replace("_vs_E2", ""),
                     v["a_better"] if k.startswith("E2") else v["b_better"], v["n"], v["wilcoxon_p"], v.get("holm_p", float("nan"))))
                    for k, v in P.items() if "E2" in k), ""]
    B = S[s].get("best_product_vs_best_qaoa")
    F = S[s].get("p_opt_near_alpha_p3")
    V = S[s].get("objective_visibility_median")
    if V:
        L += ["Objective visibility after normalisation (max |objective coefficient| / max |coefficient|, median): " +
              ", ".join("%s %.2g" % (e, v) for e, v in V.items()) + ".", ""]
    if B:
        L += ["Unentangled control: the best product-state run (any encoding, either objective) beats the best "
              "QAOA p = 3 run on %d/%d instances. Per encoding: %s." % (
                  B["product_better"], B["n"],
                  ", ".join("%s %d/%d" % (e, v["product_better"], v["n"]) for e, v in B["per_encoding"].items())), ""]
    if F:
        L += ["CVaR ceiling (proxy): runs whose P(opt) at p = 3 reached ≈ α = 0.25, beyond which CVaR₀.₂₅ gives no further reward "
              "(for E0/E1 the slack must also be right; for E3 the minimum need not be the optimum): " +
              ", ".join("%s %d/%d" % (e, v["reached"], v["n"]) for e, v in F.items()) + ".", ""]
    return "\n".join(L)


body = "\n".join(table(s) for s in ["bench", "t0", "c", "aml"] if s in S)
body += ("\nMedians over instances; 95% bootstrap CIs. *exact* = encoded minimum is the true optimum. "
         "*× random* = P(opt) divided by uniform random guessing. Full numbers (p = 1, 2, 3; both objectives; "
         "optimiser convergence) in `results/summary.json`; raw per-instance results in `results/*_raw.jsonl`.\n")
for path in ("RESULTS.md",):
    t = open(path).read()
    block = "<!-- RESULTS -->\n" + body + "<!-- /RESULTS -->"
    if "RESULTS_TABLE_PLACEHOLDER" in t:
        t = t.replace("RESULTS_TABLE_PLACEHOLDER", block)
    else:
        t = re.sub(r"<!-- RESULTS -->.*?<!-- /RESULTS -->", lambda m: block, t, flags=re.S)
    open(path, "w").write(t)
print(body)
