"""Gridlock Encodings: how constraint encoding changes QAOA on interbank settlement.

Problem: a batch of queued interbank payments is gridlocked (each bank needs
incoming money before it can pay). Choose the largest-value set of payments to
settle simultaneously so that no bank's net outflow exceeds its liquidity.
We encode the liquidity constraints five ways and run QAOA on each.

Modes (Run settings):
  demo    - one instance, one encoding: baselines, exactness check, QAOA angle
            optimisation (exact statevector), then ONE sampled run on the backend.
  compare - one instance, several encodings optimised live, then all of them packed
            side by side into ONE circuit (one job, also valid on IonQ simulators).
  results - print the shipped offline benchmark summary (80 generated instances in three
            sets, plus 22 public-data batches).

This effort is supported by Qollab & IonQ.
"""
import json
import time

import numpy as np

from settle.instances import load_instances_json
from settle.exact import brute_force, greedy, deletion_heuristic, milp, feasibility_mask, values
from settle.aml import bs_fifo
from settle.encodings import e0_standard, e1_iqp, e2_indicator, e3_unbalanced, e4_overflow_sq
from settle.verify import check_exact
from settle import poly as P
from settle import iqp
from settle.qaoa import (Problem, optimise_sweep, statevector, logical_distribution, qaoa_circuit,
                         packed_circuit, split_counts)
from settle.metrics import quality


def load_assets():
    """Instances, IQP polynomials (derived offline with scipy MILP) and E3 weights
    (tuned offline on separate training instances). Plain literal paths (Qollab rule)."""
    insts = {}
    with open("data/demo_instances.json") as f:
        insts.update({i.name: i for i in load_instances_json(f.read())})
    with open("data/bench_instances.json") as f:
        insts.update({i.name: i for i in load_instances_json(f.read())})
    with open("data/t0_instances.json") as f:
        insts.update({i.name: i for i in load_instances_json(f.read())})
    with open("data/c_instances.json") as f:
        insts.update({i.name: i for i in load_instances_json(f.read())})
    with open("data/aml_instances.json") as f:          # public-data tier (see DATA.md)
        insts.update({i.name: i for i in load_instances_json(f.read())})
    with open("settle/iqp_data/iqp_cache.json") as f:
        cache = iqp.load_cache_text(f.read())
    e3 = {}
    with open("results/e3_tuning_bench.json") as f:
        e3["bench"] = json.load(f)
    with open("results/e3_tuning_t0.json") as f:
        e3["t0"] = json.load(f)
    with open("results/e3_tuning_c.json") as f:
        e3["c"] = json.load(f)
    return insts, cache, e3


def e3_set(inst):
    """E3 weights are tuned per instance set, in that set's raw amount units: T* -> t0,
    C* -> c, AML* -> None (cents; no tuning exists, so E3 is not available), else bench."""
    if inst.name.startswith("AML"):
        return None
    return {"T": "t0", "C": "c"}.get(inst.name[0], "bench")


def _e4(inst, rule):
    if inst.name.startswith("AML"):
        raise NotAvailable("at cent scale its coefficients reach ~1e26; after normalisation the optimum and other "
                           "settlements differ by ~1e-16, below float64 resolution (correct only in exact arithmetic)")
    return e4_overflow_sq(inst, rule=rule)


class NotAvailable(Exception):
    """An encoding that does not apply to this instance (E3/E4 on the cent-scale AML tier, or E1
    when its IQP polynomial for a bank pattern was not precomputed)."""


def _e3(inst, e3):
    s = e3_set(inst)
    if s is None:
        raise NotAvailable("E3 weights were tuned in small-integer units and do not transfer to cents")
    return e3_unbalanced(inst, e3[s]["l1"], e3[s]["l2"])


def _e1(inst, rule, cache):
    try:
        return e1_iqp(inst, rule=rule, cache=dict(cache), derive_missing=False)
    except KeyError:
        raise NotAvailable("its IQP polynomial for one of this batch's bank patterns is not precomputed "
                           "(run scripts/build_iqp_cache.py offline to add it)")


def make_encodings(cache, e3):
    return {
        "E0": lambda inst, rule: e0_standard(inst, rule=rule),
        # derive_missing=False: polynomials are precomputed offline; never start a MILP in the browser
        "E1": lambda inst, rule: _e1(inst, rule, cache),
        "E2": lambda inst, rule: e2_indicator(inst, rule=rule),
        "E3": lambda inst, rule: _e3(inst, e3),
        "E4": lambda inst, rule: _e4(inst, rule),
    }


MAX_CHECK_BITS = 20      # largest per-bank enumeration (payments + slack bits) done in the browser

LABELS = {"E0": "standard binary slack (QUBO)", "E1": "IQP, few slack bits (QUBO)",
          "E2": "indicator, no slack (HUBO)", "E3": "unbalanced penalization (QUBO)",
          "E4": "overflow-squared, no slack (HUBO)"}


def fmt_x(x_idx, n):
    return "".join(str((x_idx >> k) & 1) for k in range(n))   # payment 0 first


def get_backend():
    try:
        be = backend  # noqa: F821  provided by Qollab's Select-QPU dialog
    except NameError:
        from qiskit.providers.basic_provider import BasicSimulator
        be = BasicSimulator()
    # Noise model (e.g. Aria 1 / Forte 1) is chosen in Qollab's Select-QPU dialog; we do not
    # override it in code (Qollab FAQ: set_options would override the dialog's choice).
    return be


def money(inst, v):
    """Amounts are integers: small units on the controlled tiers, US cents on the AML tier."""
    if inst.meta.get("unit") == "US cents":
        return "$%s" % format(v / 100.0, ",.2f")
    return "%d" % round(v)


def settled(inst, x):
    x = np.asarray(x)
    return "value %s of %s queued (%d of %d payments)" % (money(inst, inst.value(x)), money(inst, inst.weights.sum()),
                                                         int(x.sum()), inst.n)


def describe(inst):
    print("Instance %s: %d banks, %d queued payments" % (inst.name, inst.n_banks, inst.n))
    for k, (a, b, w) in enumerate(inst.arcs):
        print("  payment %d: bank %d -> bank %d, amount %s" % (k, a, b, money(inst, w)))
    print("  liquidity per bank:", ", ".join(money(inst, c) for c in inst.liquidity))
    if inst.meta.get("source", "").startswith("IBM AML"):
        print("  (public-data tier: USD wire payments from the IBM AML dataset, CSV rows %s; liquidity by rule f = %s)"
              % (inst.meta.get("source_rows"), inst.meta.get("f_liq")))
    gv, gx = greedy(inst)
    dv, dx = deletion_heuristic(inst)
    bv, bx = bs_fifo(inst)
    # Exact optimum by enumerating all 2^n selections (n <= 10 here: at most 1,024, a few ms).
    # The MILP (settle.exact.milp, HiGHS) gives the same optimum on every shipped instance
    # (tests/test_core.py, tests/test_aml.py); in the browser its start-up takes seconds to tens
    # of seconds, so the live demo enumerates instead.
    t = time.perf_counter()
    opt, idx = brute_force(inst)
    ox = np.array([(idx[0] >> k) & 1 for k in range(inst.n)])
    t_opt = time.perf_counter() - t
    print("\n[1] Classical baselines")
    print("  RTGS queue (greedy, largest first): settles %s" % settled(inst, gx))
    print("  Bech-Soramaki FIFO (RTGS gridlock): settles %s" % settled(inst, bx))
    print("  Deletion heuristic (multilateral):  settles %s" % settled(inst, dx))
    print("  Exact optimum (all %d selections, %.3fs): settles %s, x = %s (payment 0 first)"
          % (2 ** inst.n, t_opt, settled(inst, ox), "".join(map(str, ox))))
    return gv, opt


def resources(inst, encs, tuning):
    print("\n[2] Encodings: qubits (payments + slack), Ising terms by order, CNOTs per QAOA layer, exact?")
    for name, mk in encs.items():
        try:
            e = mk(inst, tuning)
        except NotAvailable as err:
            print("  %s %-37s n/a: %s" % (name, LABELS[name], err))
            continue
        st = P.term_stats(P.to_ising(e.poly))
        # The exactness check enumerates each bank's payment + slack bits. With cent-scale
        # amounts E0 has ~30 slack bits per bank (2^35 states): too much for a browser.
        if max((len(b.xvars) + len(b.svars) for b in e.blocks), default=0) > MAX_CHECK_BITS:
            verdict = "exactness not checked here (too many slack bits per bank)"
        else:
            verdict = "exact" if check_exact(inst, e)["exact"] else "NOT exact (ground state is not the optimum)"
        print("  %s %-37s %3d qubits (%3d slack)  terms %-28s ~%3d CNOT  %s"
              % (name, LABELS[name], e.n_total, e.n_slack, st["terms_by_order"], st["cnot_ladder"], verdict))


def run_demo(inst, encs, encoding, p, shots, tuning, objective, alpha):
    gv, opt = describe(inst)
    resources(inst, encs, tuning)
    try:
        enc = encs[encoding](inst, tuning)
    except NotAvailable as err:
        print("\n%s is not available for %s: %s. Pick another encoding." % (encoding, inst.name, err))
        return
    if enc.n_total > 20:
        print("\n%s needs %d qubits: too many for the in-browser optimiser. Pick a smaller "
              "instance (e.g. A18) or another encoding." % (encoding, enc.n_total))
        return
    prob = Problem(enc)
    print("\n[3] Optimising QAOA angles for %s (exact statevector, p = 1..%d, objective %s)"
          % (encoding, p, "CVaR_%g" % alpha if objective == "cvar" else "<H>"))
    sweep = optimise_sweep(prob, p, objective=objective, alpha=alpha)
    for pp, (E, g, b, ev, conv) in sweep.items():
        q = quality(inst, logical_distribution(prob, statevector(prob, g, b)))
        print("  p=%d  objective=%.3f  P(optimum)=%.3f  P(feasible)=%.3f  (%d evaluations)"
              % (pp, E, q["p_opt"], q["p_feasible"], ev))
    E, g, b, _, _ = sweep[p]
    q_exact = quality(inst, logical_distribution(prob, statevector(prob, g, b)))

    print("\n[4] Sampling the circuit once on the backend")
    be = get_backend()
    from qiskit import transpile
    tq = transpile(qaoa_circuit(prob, g, b), backend=be, optimization_level=1)
    print("  backend: %s | %d qubits | depth %d" % (getattr(be, "name", be), tq.num_qubits, tq.depth()))
    counts = be.run(tq, shots=shots).result().get_counts()
    px = split_counts(counts, [(prob, g, b)], [0])[0]
    report_samples(inst, px, q_exact, gv, opt, shots)


def report_samples(inst, px, q_exact, gv, opt, shots, label=""):
    """px: sampled frequencies. Compared with the exact (statevector) value and with
    uniform random guessing at the same number of shots."""
    qs = quality(inst, px)
    hits = int(round(qs["p_opt"] * shots))
    pu = qs["uniform_p_opt"]
    print("  %sP(optimum):  %d shots %.3f | exact %.3f | uniform random %.4f  -> %.1fx random"
          % (label, shots, qs["p_opt"], q_exact["p_opt"], pu, qs["p_opt"] / pu))
    print("  %sP(feasible): %d shots %.3f | exact %.3f | uniform random %.4f"
          % (label, shots, qs["p_feasible"], q_exact["p_feasible"], qs["uniform_p_feasible"]))
    # Expected shots to the first optimum = 1 / P(opt). Use the frequency actually sampled on
    # this backend (on a noisy backend it differs from the exact noiseless value).
    if qs["p_opt"] > 0:
        print("  %sexpected shots to first optimum: %.0f from this run's samples (exact noiseless: %s; "
              "uniform random: %.0f); optimum seen %d times in %d shots"
              % (label, 1 / qs["p_opt"], "%.0f" % (1 / q_exact["p_opt"]) if q_exact["p_opt"] > 0 else "n/a",
                 1 / pu, hits, shots))
    else:
        print("  %soptimum not seen in %d shots (exact noiseless P(opt) %.3f; uniform random needs ~%.0f shots)"
              % (label, shots, q_exact["p_opt"], 1 / pu))


def run_compare(inst, encs, p, shots, tuning, objective, alpha, max_qubits):
    gv, opt = describe(inst)
    resources(inst, encs, tuning)
    be = get_backend()
    on_ionq = str(getattr(be, "name", "")).startswith("ionq")
    print("\n[3] Optimising each encoding (p=%d, %s)" % (p, "CVaR_%g" % alpha if objective == "cvar" else "<H>"))
    blocks, names, exact_q, used = [], [], [], 0
    for name in ("E1", "E2", "E3", "E4", "E0"):
        try:
            enc = encs[name](inst, tuning)
        except NotAvailable:
            print("  %s not available for this instance" % name)
            continue
        too_big = enc.n_total > 18 or (on_ionq and used + enc.n_total > max_qubits)
        if too_big:
            print("  %s skipped (%d qubits)" % (name, enc.n_total))
            continue
        prob = Problem(enc)
        E, g, b, ev, _ = optimise_sweep(prob, p, objective=objective, alpha=alpha)[p]
        q = quality(inst, logical_distribution(prob, statevector(prob, g, b)))
        print("  %s %2d qubits  P(optimum)=%.3f  P(feasible)=%.3f" % (name, enc.n_total, q["p_opt"], q["p_feasible"]))
        blocks.append((prob, g, b)); names.append(name); exact_q.append(q); used += enc.n_total
    from qiskit import transpile
    if on_ionq:
        # IonQ: one job per run, one circuit per job -> all encodings side by side in one circuit
        print("\n[4] ONE circuit with blocks %s (%d qubits), one IonQ job on backend %s"
              % (names, used, getattr(be, "name", be)))
        qc, offs = packed_circuit(blocks)
        tq = transpile(qc, backend=be, optimization_level=1)
        counts = be.run(tq, shots=shots).result().get_counts()
        dists = split_counts(counts, blocks, offs)
    else:
        print("\n[4] Sampling each encoding on %s" % getattr(be, "name", be))
        dists = []
        for blk in blocks:
            tq = transpile(qaoa_circuit(blk[0], blk[1], blk[2]), backend=be, optimization_level=1)
            counts = be.run(tq, shots=shots).result().get_counts()
            dists.append(split_counts(counts, [blk], [0])[0])
    for name, px, q in zip(names, dists, exact_q):
        print(" ", name, LABELS[name])
        report_samples(inst, px, q, gv, opt, shots, label="    ")


SET_TITLES = {"bench": "Main set: 40 instances, 4-5 banks, 6-8 payments",
              "t0": "Tier T0: 20 instances, small amounts (E0 fits in 18 qubits)",
              "c": "Tier C: 20 instances, 6 banks, 10 payments",
              "aml": "Public data (IBM AML): 22 gridlocked batches, exact cents"}


def bar(v, width=40):
    """Text bar on a log scale from 0.001 to 1 (P(opt) values span three decades)."""
    if v <= 0:
        return ""
    x = (np.log10(max(v, 1e-3)) + 3) / 3
    return "#" * max(1, int(round(x * width)))


def run_results():
    """Shipped offline benchmark, drawn as text (Qollab shows the console, not image or JSON files)."""
    try:
        with open("results/summary.json") as f:
            S = json.load(f)
    except OSError:
        print("results/summary.json not shipped yet.")
        return
    print("Offline benchmark: probability that one QAOA sample is the optimal settlement.")
    print("QAOA p = 3, CVaR_0.25 objective, median over instances. Bars: log scale, 0.001 to 1.")
    print("'x rand' = times better than uniform random guessing. 'product' = unentangled control (<H>).")
    for s in ("bench", "t0", "c", "aml"):
        if s not in S:
            continue
        d = S[s]["by_encoding"]
        print("\n== %s ==" % SET_TITLES[s])
        for e in ("E0", "E1", "E2", "E3", "E4"):
            k = "%s|cvar" % e
            if k not in d:
                why = {("E3", "aml"): "n/a (not tuned for cents)",
                       ("E4", "aml"): "n/a (float64 limit at cents)"}.get((e, s), "not simulated (needs > 18 qubits)")
                print("  %s %-8s %s" % (e, "", why))
                continue
            x, m = d[k], d["%s|mean" % e]
            po = x["p"]["3"]["p_opt"][0]
            print("  %s %-6.3f %-40s %5.1fx rand | product %.2f | exact %d/%d | n=%d"
                  % (e, po, bar(po), x["p"]["3"]["enrichment"][0], m["product"]["p_opt"][0], x["exact"], x["n"], x["n"]))
        B = S[s].get("best_product_vs_best_qaoa")
        if B:
            print("  best unentangled run beats best QAOA run on %d/%d instances" % (B["product_better"], B["n"]))
    print("\nFull tables: RESULTS.md and results/summary.json. Encodings: E0 standard slack, E1 IQP, "
          "E2 indicator, E3 unbalanced, E4 overflow-squared.")


def main(mode: str = "demo", instance: str = "AML043", encoding: str = "E2", p: int = 2, shots: int = 1000, objective: str = "cvar", alpha: float = 0.25, tuning: str = "neighbour", max_qubits: int = 25):
    t0 = time.perf_counter()
    insts, cache, e3 = load_assets()
    encs = make_encodings(cache, e3)
    # Run-settings fields are typed by hand: tolerate spaces and letter case, and say what was received.
    mode = str(mode).strip().lower()
    encoding = str(encoding).strip().upper()
    objective = str(objective).strip().lower()
    tuning = str(tuning).strip().lower().replace("neighbor", "neighbour")
    by_lower = {k.lower(): k for k in insts}
    instance = by_lower.get(str(instance).strip().lower(), str(instance).strip())
    problems = []
    if mode not in ("demo", "compare", "results"):
        problems.append("mode %r: use demo, compare or results" % mode)
    if mode != "results":
        if instance not in insts:
            problems.append("instance %r: not found; examples: AML043, gridlock6x9, A19, T03, C00 (all names in data/*.json)" % instance)
        if mode == "demo" and encoding not in encs:
            problems.append("encoding %r: use E0, E1, E2, E3 or E4" % encoding)
        if objective not in ("cvar", "mean"):
            problems.append("objective %r: use cvar or mean" % objective)
        if tuning not in ("local", "neighbour", "global"):
            problems.append("tuning %r: use local, neighbour or global" % tuning)
        if int(p) < 1 or int(shots) < 1 or not (0 < float(alpha) <= 1):
            problems.append("p and shots must be >= 1 and alpha in (0, 1]")
    if problems:
        print("Run settings not understood:")
        for msg in problems:
            print("  - " + msg)
    elif mode == "results":
        run_results()
    elif mode == "demo":
        run_demo(insts[instance], encs, encoding, int(p), int(shots), tuning, objective, float(alpha))
    else:
        run_compare(insts[instance], encs, int(p), int(shots), tuning, objective, float(alpha), int(max_qubits))
    print("\nDone in %.1fs. This effort is supported by Qollab & IonQ." % (time.perf_counter() - t0))
