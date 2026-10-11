"""Noise check: do the encoding results survive hardware-like noise?

Re-evaluates the SHIPPED p = 3 CVaR angles (results/<set>_raw.jsonl, optimised
noiselessly) under a depolarising noise model with trapped-ion-like error rates.
No re-optimisation, no new tuning. All-to-all connectivity is assumed (as on
IonQ hardware), so no SWAPs: the two-qubit gate count is CX + RZZ in the circuit.

Compiled variants (two-qubit gate = CX or native RZZ; IonQ implements a ZZ rotation
as one native gate: partial-angle MS on Aria, ZZ on Forte):
  E1          shipped circuit: 2 CX per ZZ term
  E1-zz       same unitary, 1 native RZZ per ZZ term
  E2          shipped circuit: CX ladder per term, 2(k-1) CX for a k-body term
  E2-gray     same unitary; per bank, Gray-code order (Welch et al. 2014,
              at most 2^d - 2 CX) where cheaper than the CX ladder
  E2-best     same unitary; per bank, the cheaper of Gray code and a ladder with
              a native RZZ core (2(k-2)+1 two-qubit gates per k-body term)
Fair comparisons: E1 vs E2 (both naive) and E1-zz vs E2-best (both native).
Gate ORDER differs between variants even where counts are equal, and under noise
order alone can change P(opt) slightly (~1e-3).

Checks before any noisy number is reported:
  * noiseless density-matrix P(opt) equals the shipped P(opt) (|diff| < 1e-6)
  * every compiled variant's noiseless P(opt) equals its shipped circuit's (|diff| < 1e-9)

Noise model (per gate, depolarising): two-qubit parameter e on every CX and RZZ,
one-qubit parameter e/5 on H and RX (RZ is virtual: no error), symmetric readout
error r = 0.005 on each measured qubit; e swept over 0.002, 0.005, 0.01.
Average gate fidelity is 1 - 0.75e (two-qubit) and 1 - 0.5(e/5) (one-qubit), so
e = 0.005 gives 99.625% / 99.95%, matching IonQ Aria as listed on Microsoft's
Azure Quantum provider page (99.6% two-qubit, 99.95% one-qubit, 99.61% SPAM).
This is a generic depolarising model, NOT IonQ's own noise model, and the angles
were optimised without noise.

Usage (offline; needs qiskit-aer, ~15 min on 2 cores):
  python3 scripts/noise_check.py bench aml   -> results/noise_check.json
Paired statistics: wins with a 1e-9 tolerance, two-sided Wilcoxon, unadjusted.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from settle.instances import load_instances_json
from settle.encodings import e1_iqp, e2_indicator
from settle.qaoa import Problem
from settle.metrics import quality
from settle import poly as P
from settle import iqp

E2_LEVELS = [0.002, 0.005, 0.01]
READOUT = 0.005
FILES = {"bench": "bench_instances.json", "aml": "aml_instances.json", "t0": "t0_instances.json"}
CACHE = None


def ladder(qc, terms, gamma, zz=False):
    """CNOT ladder per term. zz=True: the last link is one native RZZ (IonQ: partial-angle
    MS on Aria, ZZ on Forte), so a k-body term costs 2(k-2)+1 two-qubit gates instead of 2(k-1)."""
    cn = 0
    for t, c in sorted(terms.items(), key=lambda kv: (len(kv[0]), kv[0])):
        if len(t) == 1:
            qc.rz(2 * gamma * c, t[0]); continue
        if zz:
            for a, b in zip(t[:-2], t[1:-1]):
                qc.cx(a, b); cn += 1
            qc.rzz(2 * gamma * c, t[-2], t[-1]); cn += 1
            for a, b in reversed(list(zip(t[:-2], t[1:-1]))):
                qc.cx(a, b); cn += 1
            continue
        for a, b in zip(t[:-1], t[1:]):
            qc.cx(a, b); cn += 1
        qc.rz(2 * gamma * c, t[-1])
        for a, b in reversed(list(zip(t[:-1], t[1:]))):
            qc.cx(a, b); cn += 1
    return cn


def gray(qc, terms, qubits, gamma):
    """All terms lie inside `qubits`; Gray-code traversal per pivot qubit."""
    d = len(qubits)
    pos = {q: j for j, q in enumerate(qubits)}
    coef = {frozenset(pos[q] for q in t): c for t, c in terms.items()}
    cn = 0
    for i in range(d):
        c = coef.get(frozenset({i}), 0.0)
        if c:
            qc.rz(2 * gamma * c, qubits[i])
        if i == 0:
            continue
        cur, prev = set(), 0
        for g in range(1, 2 ** i):
            gc = g ^ (g >> 1)
            flip = (gc ^ prev).bit_length() - 1
            prev = gc
            qc.cx(qubits[flip], qubits[i]); cn += 1
            cur ^= {flip}
            c = coef.get(frozenset(cur | {i}), 0.0)
            if c:
                qc.rz(2 * gamma * c, qubits[i])
        qc.cx(qubits[prev.bit_length() - 1], qubits[i]); cn += 1
    return cn


def split_by_bank(prob):
    """Assign each multi-body normalised term to the first bank containing it."""
    banks = [sorted(b.xvars) for b in prob.enc.blocks]
    per = [dict() for _ in banks]
    rest = {}
    for t, c in prob.zterms.items():
        if len(t) >= 2:
            for j, vs in enumerate(banks):
                if set(t) <= set(vs):
                    per[j][t] = c
                    break
            else:
                rest[t] = c
        else:
            rest[t] = c
    return banks, per, rest


def cost_layer(qc, prob, gamma, mode):
    """ladder: shipped CNOT ladder.  zz: ladder with a native RZZ core.
    gray: per bank, Gray code where cheaper than the CNOT ladder.
    best: per bank, the cheaper of Gray code and the RZZ-core ladder."""
    if mode in ("ladder", "zz"):
        return ladder(qc, prob.zterms, gamma, zz=(mode == "zz"))
    zz = mode == "best"
    banks, per, rest = split_by_bank(prob)
    cn = ladder(qc, rest, gamma, zz=zz)
    for vs, terms in zip(banks, per):
        if not terms:
            continue
        lad_cost = sum((2 * (len(t) - 2) + 1) if zz else 2 * (len(t) - 1) for t in terms)
        if 2 ** len(vs) - 2 < lad_cost:
            cn += gray(qc, terms, vs, gamma)
        else:
            cn += ladder(qc, terms, gamma, zz=zz)
    return cn


def circuit(prob, gammas, betas, mode):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(prob.n)
    qc.h(range(prob.n))
    cn = 0
    for g, b in zip(gammas, betas):
        cn += cost_layer(qc, prob, g, mode)
        qc.rx(2 * b, range(prob.n))
    return qc, cn


def probs(qc, e2):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    qc = qc.copy()
    qc.save_density_matrix()
    if e2 > 0:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ["cx", "rzz"])
        nm.add_all_qubit_quantum_error(depolarizing_error(e2 / 5, 1), ["h", "rx"])
        sim = AerSimulator(method="density_matrix", noise_model=nm)
    else:
        sim = AerSimulator(method="density_matrix")
    res = sim.run(qc, optimization_level=0).result()
    rho = np.asarray(res.data(0)["density_matrix"])
    return np.real(np.diag(rho)).copy()


def readout(p, n, r):
    p = p.copy()
    for q in range(n):
        v = p.reshape(-1, 2, 2 ** q)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = (1 - r) * a + r * b
        v[:, 1, :] = r * a + (1 - r) * b
    return p


def logical(p, prob):
    return p.reshape(-1, 2 ** prob.n_logical).sum(axis=0)


def job(args):
    set_name, inst_d, enc_name, gammas, betas, shipped = args
    import qiskit_aer  # noqa: F401  (fail early in workers)
    inst = load_instances_json(json.dumps([inst_d]))[0]
    enc = e1_iqp(inst, rule="neighbour", cache=dict(CACHE), derive_missing=False) if enc_name == "E1" \
        else e2_indicator(inst, rule="neighbour")
    prob = Problem(enc)
    modes = ["ladder", "zz"] if enc_name == "E1" else ["ladder", "gray", "best"]
    out = []
    for mode in modes:
        qc, cn = circuit(prob, gammas, betas, mode)
        p0 = quality(inst, logical(probs(qc, 0.0), prob))["p_opt"]
        row = {"set": set_name, "instance": inst.name, "variant": enc_name + ("" if mode == "ladder" else "-" + mode),
               "n_qubits": prob.n, "twoq_total_p3": cn, "p_opt_noiseless": p0, "p_opt_shipped": shipped,
               "uniform_p_opt": quality(inst, np.full(2 ** inst.n, 1 / 2 ** inst.n))["p_opt"], "noisy": {}}
        for e2 in E2_LEVELS:
            p = readout(probs(qc, e2), prob.n, READOUT)
            row["noisy"][str(e2)] = quality(inst, logical(p, prob))["p_opt"]
        out.append(row)
    return out


def main(sets):
    global CACHE
    CACHE = iqp.load_cache_text(open(os.path.join(ROOT, "settle/iqp_data/iqp_cache.json")).read())
    tasks = []
    for s in sets:
        insts = {d["name"]: d for d in json.load(open(os.path.join(ROOT, "data", FILES[s])))}
        for line in open(os.path.join(ROOT, "results", "%s_raw.jsonl" % s)):
            r = json.loads(line)
            if r["objective"] != "cvar" or r["encoding"] not in ("E1", "E2"):
                continue
            q = r["qaoa"]["3"]
            tasks.append((s, insts[r["instance"]], r["encoding"], q["gammas"], q["betas"], q["p_opt"]))
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(initializer=_init, initargs=(CACHE,)) as ex:
        for out in ex.map(job, tasks):
            rows.extend(out)
    # checks
    bad = [r for r in rows if abs(r["p_opt_noiseless"] - r["p_opt_shipped"]) > 1e-6]
    pair = {}
    for r in rows:
        pair.setdefault((r["set"], r["instance"]), {})[r["variant"]] = r
    bad_gray = [(k, var) for k, v in pair.items() for var in ("E1-zz", "E2-gray", "E2-best") if var in v and
                abs(v[var]["p_opt_noiseless"] - v[var.split("-")[0]]["p_opt_noiseless"]) > 1e-9]
    print("rows %d, %.0fs; noiseless != shipped: %d; compiled variant != shipped circuit: %d" % (len(rows), time.time() - t0, len(bad), len(bad_gray)))
    assert not bad and not bad_gray, (bad[:2], bad_gray[:2])
    summary = {}
    for s in sets:
        for v in ["E1", "E1-zz", "E2", "E2-gray", "E2-best"]:
            R = [r for r in rows if r["set"] == s and r["variant"] == v]
            if not R:
                continue
            d = {"n": len(R), "twoq_p3_median": float(np.median([r["twoq_total_p3"] for r in R])),
                 "p_opt_noiseless_median": float(np.median([r["p_opt_noiseless"] for r in R]))}
            for e2 in E2_LEVELS:
                vals = [r["noisy"][str(e2)] for r in R]
                d["p_opt_median_e2=%s" % e2] = float(np.median(vals))
                d["x_random_median_e2=%s" % e2] = float(np.median([r["noisy"][str(e2)] / r["uniform_p_opt"] for r in R]))
            summary["%s/%s" % (s, v)] = d
        # paired comparisons per noise level: wins counted with a 1e-9 tolerance (ties are not
        # wins); two-sided Wilcoxon signed-rank p, unadjusted (zero differences dropped)
        from scipy.stats import wilcoxon
        ks = [k for k in pair if k[0] == s and {"E1", "E1-zz", "E2", "E2-gray", "E2-best"} <= set(pair[k])]
        for e2 in [0.0] + E2_LEVELS:
            key = (lambda r: r["p_opt_noiseless"]) if e2 == 0 else (lambda r, e=str(e2): r["noisy"][e])
            d = {"n": len(ks)}
            for a, b in [("E2", "E1"), ("E2-best", "E1-zz"), ("E2-gray", "E1"), ("E2-gray", "E1-zz"), ("E2", "E1-zz"), ("E2-best", "E2")]:
                x = np.array([key(pair[k][a]) for k in ks])
                y = np.array([key(pair[k][b]) for k in ks])
                diff = x - y
                nz = np.abs(diff) > 1e-9
                p = float(wilcoxon(x[nz], y[nz]).pvalue) if nz.sum() >= 5 else None
                d["%s_vs_%s" % (a, b)] = {"wins": int((diff > 1e-9).sum()), "losses": int((diff < -1e-9).sum()),
                                          "ties": int((~nz).sum()), "wilcoxon_p": p}
            summary["%s/paired_e2=%s" % (s, e2)] = d
    meta = {"model": "depolarising: e2 on CX and RZZ, e2/5 on H and RX, RZ virtual; readout %g per qubit" % READOUT,
            "angles": "shipped p=3 CVaR angles, no re-optimisation", "levels_e2": E2_LEVELS}
    json.dump({"meta": meta, "summary": summary, "rows": rows},
              open(os.path.join(ROOT, "results", "noise_check.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


def _init(cache):
    global CACHE
    CACHE = cache


if __name__ == "__main__":
    main(sys.argv[1:] or ["bench", "aml"])
