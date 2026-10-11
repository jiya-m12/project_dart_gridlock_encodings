"""Figure for README/video: P(optimum) per encoding, three instance sets.
QAOA p=3 (CVaR_0.25, its better objective) median with 95% bootstrap CI, versus the
unentangled product-state control (<H>, its better objective) and uniform random
guessing. Log scale. Output: results/fig_popt.png
Width is capped at 2000 px (Qollab project-card images: max 2048 x 2048). A footnote records
where the numbers come from (protocol tag and results/run_env_*.json), as the Qollab FAQ asks."""
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("results/summary.json"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
QAOA_C, PROD_C, RAND_C = "#2a78d6", "#eb6834", "#8a8984"
NAMES = {"E0": "E0 standard slack (QUBO)", "E1": "E1 IQP (QUBO)", "E2": "E2 indicator (HUBO)",
         "E3": "E3 unbalanced (QUBO)", "E4": "E4 overflow² (HUBO)"}
TITLES = {"bench": "Main set: 40 instances\n4–5 banks, 6–8 payments",
          "t0": "Tier T0: 20 instances\n4 banks, 6 payments, small amounts",
          "c": "Tier C: 20 instances\n6 banks, 10 payments",
          "aml": "Public data (IBM AML): 22 batches\n4–5 banks, 6–9 payments, cents"}
raw = {s: [json.loads(l) for l in open("results/%s_raw.jsonl" % s)] for s in S}

SETS = [s for s in ["bench", "t0", "c", "aml"] if s in S]
FIG_W = 4.4 * len(SETS) + 0.4
fig, axes = plt.subplots(1, len(SETS), figsize=(FIG_W, 5.0), sharey=True, facecolor=SURF)
axes = np.atleast_1d(axes)
encs = ["E0", "E1", "E2", "E3", "E4"]
XMIN = 4e-4
for ax, s in zip(axes, SETS):
    ax.set_facecolor(SURF)
    d = S[s]["by_encoding"]
    unif = np.median([r["qaoa"]["3"]["uniform_p_opt"] for r in raw[s] if r["objective"] == "cvar"])
    ax.axvline(unif, color=RAND_C, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(unif * 1.1, -0.45, "uniform random", color=INK2, fontsize=8, va="center")
    for y, e in enumerate(encs):
        kq, kp = "%s|cvar" % e, "%s|mean" % e
        if kq not in d or d[kq]["n"] < 5:
            if kq not in d:
                msg = {("E3", "aml"): "n/a (E3 not tuned for cents)",
                       ("E4", "aml"): "n/a (float64 limit at cents)"}.get((e, s), "not simulated (>18 qubits)")
            else:
                msg = "only %d instance fits in 18 qubits (see T0)" % d[kq]["n"]
            ax.text(0.03, y, msg, color=INK2, fontsize=8, va="center")
            continue
        m, lo, hi = d[kq]["p"]["3"]["p_opt"]
        ax.plot([lo, hi], [y, y], color=QAOA_C, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(m, y, "o", ms=8, color=QAOA_C, mec=SURF, mew=2, zorder=3)
        pm, plo, phi = d[kp]["product"]["p_opt"]
        if pm < 1e-3:   # log axis cannot show ~0: pin to the floor and say so
            ax.plot(XMIN * 1.15, y + 0.22, "D", ms=7, color=PROD_C, mec=SURF, mew=2, zorder=3, clip_on=False)
            ax.text(XMIN * 1.5, y + 0.22, "~0", color=INK2, fontsize=8, va="center")
        else:
            ax.plot(pm, y + 0.22, "D", ms=7, color=PROD_C, mec=SURF, mew=2, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(XMIN, 1.5)
    ax.set_ylim(-0.75, 4.6)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_title(TITLES[s], color=INK, fontsize=10, loc="left")
    ax.set_xlabel("P(optimum), log scale", color=INK2, fontsize=9)
axes[0].set_yticks(range(5))
axes[0].set_yticklabels([NAMES[e] for e in encs], color=INK, fontsize=9)
h1 = plt.Line2D([], [], color=QAOA_C, marker="o", ms=8, lw=2, ls="-", mec=SURF, mew=2,
                label="QAOA p=3, CVaR 0.25: median and 95% CI")
h2 = plt.Line2D([], [], color=PROD_C, marker="D", ms=7, lw=0, mec=SURF, mew=2,
                label="Product state, no entanglement (<H>): median")
h3 = plt.Line2D([], [], color=RAND_C, lw=1.5, ls=(0, (4, 3)), label="Uniform random guessing")
fig.legend(handles=[h1, h2, h3], loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
           bbox_to_anchor=(0.5, 0.075))
protocols = sorted({r.get("protocol", "?") for rows in raw.values() for r in rows})
envs = []
for path in sorted(glob.glob("results/run_env_*.json")):
    e = json.load(open(path))
    envs.append("%s: %s, Python %s, numpy %s, scipy %s, started %s" % (
        "+".join(e.get("sets", [])), e.get("machine", "?"), e.get("python", "?"), e.get("numpy", "?"),
        e.get("scipy", "?"), e.get("started", "?")))
note = ("Offline exact-statevector simulation (scripts/run_benchmark.py), protocol: %s. Medians over instances. "
        % "; ".join(protocols))
if "aml" in SETS:
    note += ("AML panel: E3 and E4 not run (E3's penalty parameters were tuned on integer amounts only; at cents "
             "E4's objective gaps fall below float64 resolution, METHOD.md §3).")
fig.text(0.01, 0.005, note + ("\nRuns: " + " | ".join(envs) if envs else ""), ha="left", va="bottom",
         fontsize=7.5, color=INK2, wrap=True)
fig.suptitle("Probability of sampling the optimal settlement, by constraint encoding", color=INK,
             fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.15, 1, 0.95))
fig.savefig("results/fig_popt.png", dpi=min(160, 2000.0 / FIG_W), facecolor=SURF)
print("saved results/fig_popt.png")
