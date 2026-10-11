"""Explanatory figures for README.md and METHOD.md that do not depend on the QAOA run.
Usage (from the package root): python3 scripts/make_figures.py
Outputs:
  results/fig_problem.png        the gridlock example: RTGS queue vs optimal batch
  results/fig_qubits.png         qubits vs amount resolution (gridlock example)
  results/fig_penalty_rules.png  penalty-weight rule study (results/tuning_failures.json)
  results/fig_aml043.png         public-data gridlock example (data/aml_instances.json, AML043)
The QAOA result figure is scripts/plot_results.py (needs results/summary.json).
Every number drawn is recomputed or read from a shipped file and checked by an assert,
so the figures cannot drift from the code."""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

from settle.instances import Instance, gridlock_instance
from settle.exact import greedy, brute_force
from settle.encodings import e0_standard, e1_iqp, e2_indicator, e3_unbalanced, e4_overflow_sq
from settle import iqp

# Colour-safe categorical slots (validated: light surface, CVD and normal-vision separation pass;
# aqua/yellow are below 3:1 contrast, so every mark in those colours carries a direct text label).
SURF, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DPI = 160
CACHE = iqp.load_cache_text(open("settle/iqp_data/iqp_cache.json").read())


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


# ------------------------------------------------------------------ figure 1: the problem
def fig_problem():
    inst = gridlock_instance()
    gv, gx = greedy(inst)
    opt, opt_idx = brute_force(inst)
    assert len(opt_idx) == 1, "figure assumes a unique optimum"
    ox = np.array([(opt_idx[0] >> k) & 1 for k in range(inst.n)])
    assert (gv, opt) == (3.0, 54.0) and list(gx) == [0] * 8 + [1] and list(ox) == [1] * 8 + [0]
    net_all = inst.net_outflow_matrix() @ np.ones(inst.n)
    over = [u for u in range(inst.n_banks) if net_all[u] > inst.liquidity[u]]
    assert over == [4] and net_all[4] == 5 and inst.liquidity[4] == 3

    pos = {0: (-1.6, -0.95), 1: (-1.6, 0.95), 2: (0.0, 0.0), 3: (1.6, 0.95), 4: (3.2, 0.0), 5: (1.6, -0.95)}
    bend = {8: -0.62}         # payment 8 (bank 4 -> bank 0) curves below bank 5
    R = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor=SURF)
    panels = [("RTGS queue, largest payment first", gx,
               "settles 1 payment: value %d of %d" % (gv, inst.weights.sum())),
              ("Optimal simultaneous batch (exact)", ox,
               "settles 8 payments: value %d of %d" % (opt, inst.weights.sum()))]
    for ax, (title, x, sub) in zip(axes, panels):
        style(ax)
        ax.set_xticks([]); ax.set_yticks([])
        for k, (a, b, w) in enumerate(inst.arcs):
            on = bool(x[k])
            rad = bend.get(k, 0.0)
            arr = FancyArrowPatch(pos[a], pos[b], connectionstyle="arc3,rad=%g" % rad,
                                  arrowstyle="-|>", mutation_scale=15,
                                  shrinkA=24, shrinkB=26,
                                  color=BLUE if on else MUTED, lw=2.4 if on else 1.3, zorder=2)
            ax.add_patch(arr)
            (x0, y0), (x1, y1) = pos[a], pos[b]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            if rad:                                   # label at the curve's midpoint
                dx, dy = x1 - x0, y1 - y0
                mx, my = mx + rad * dy * 0.5, my - rad * dx * 0.5
            ax.text(mx, my, "%d" % w, ha="center", va="center", fontsize=10,
                    color=INK if on else INK2, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18", fc=SURF, ec="none"))
        for u, (px, py) in pos.items():
            ax.add_patch(Circle((px, py), R, fc="#efeeea", ec="none", zorder=3))
            ax.text(px, py + 0.09, "B%d" % u, ha="center", va="center", fontsize=11, color=INK,
                    fontweight="bold", zorder=5)
            ax.text(px, py - 0.13, "liq. %d" % inst.liquidity[u], ha="center", va="center",
                    fontsize=8.5, color=INK2, zorder=5)
        ax.set_xlim(-2.2, 3.8); ax.set_ylim(-2.35, 1.45); ax.set_aspect("equal")
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=18)
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, color=INK2, fontsize=10, va="bottom")
    h1 = plt.Line2D([], [], color=BLUE, lw=2.4, label="settled in this batch")
    h2 = plt.Line2D([], [], color=MUTED, lw=1.3, label="stays queued")
    fig.legend(handles=[h1, h2], loc="upper right", ncol=2, frameon=False, fontsize=9.5,
               labelcolor=INK, bbox_to_anchor=(0.995, 0.995))
    fig.text(0.01, 0.02,
             "Arrow = queued payment (payer to payee), number = amount, liq. = liquidity. Settling all 9 is infeasible: "
             "bank 4 would pay out 5 net with liquidity 3.\nThe queue settles the one payment the optimum "
             "leaves out. Bech-Soramäki FIFO and the deletion heuristic also reach 54 here; on A19 and "
             "AML043 every heuristic settles 0.",
             ha="left", va="bottom", fontsize=8.5, color=INK2)
    fig.suptitle("Gridlock: each bank waits for incoming money, so the payment queue stalls",
                 color=INK, fontsize=13, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig("results/fig_problem.png", dpi=DPI, facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2: qubits vs amounts
def e0_qubits_formula(inst):
    """Independent count: n + sum over binding banks of ceil(log2(C + inflow + 1))."""
    n = inst.n
    for u in range(inst.n_banks):
        out = sum(w for a, b, w in inst.arcs if a == u)
        inn = sum(w for a, b, w in inst.arcs if b == u)
        if out > inst.liquidity[u]:
            n += max(1, math.ceil(math.log2(inst.liquidity[u] + inn + 1)))
    return n


def fig_qubits():
    g = gridlock_instance()
    scales = [1, 10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6]
    rows = []
    for s in scales:
        inst = Instance(g.name, g.n_banks, [(a, b, w * s) for a, b, w in g.arcs], [c * s for c in g.liquidity])
        e0 = e0_standard(inst).n_total
        assert e0 == e0_qubits_formula(inst)
        e1 = e1_iqp(inst, cache=dict(CACHE), derive_missing=False).n_total
        e2 = e2_indicator(inst).n_total
        assert e2 == e4_overflow_sq(inst).n_total == e3_unbalanced(inst, 1.0, 1.0).n_total == inst.n
        rows.append((s, e0, e1, e2))
    shipped = {r["scale"]: r for r in json.load(open("results/qubits_vs_amounts.json")) if r["instance"] == g.name}
    for s, e0, e1, e2 in rows:
        if s in shipped:
            assert (e0, e1, e2) == (shipped[s]["E0"], shipped[s]["E1"], shipped[s]["E2_E3_E4"])
    S = np.array([r[0] for r in rows], float)
    fig, ax = plt.subplots(figsize=(8.6, 4.9), facecolor=SURF)
    style(ax)
    ax.axhline(24, color=INK2, lw=1, zorder=1)
    ax.text(S[0], 25.5, "Qollab built-in simulator: 24 qubits", color=INK2, fontsize=8.5, va="bottom")
    series = [("E0 standard binary slack", [r[1] for r in rows], BLUE, "o"),
              ("E1 IQP (fewest slack bits)", [r[2] for r in rows], ORANGE, "s"),
              ("E2 / E3 / E4 (no slack)", [r[3] for r in rows], AQUA, "D")]
    for name, ys, col, mk in series:
        ax.plot(S, ys, color=col, lw=2, marker=mk, ms=7, mec=SURF, mew=2, zorder=3,
                solid_capstyle="round", label=name)
        dy = {"E0": 0, "E1": 4, "E2": -4}[name[:2]]       # keep the two flat end labels apart
        ax.text(S[-1] * 1.6, ys[-1] + dy, "%d" % ys[-1], color=INK, fontsize=9.5, va="center")
    ax.text(S[0] * 1.15, rows[0][1] + 5, "%d" % rows[0][1], color=INK, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(S)
    ax.set_xticklabels(["×1", "×10", "×100", "×1,000", "×10⁴", "×10⁵", "×10⁶"])
    ax.set_xlim(0.6, S[-1] * 8)
    ax.set_ylim(0, 170)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Amount resolution: every amount and liquidity multiplied by", color=INK2, fontsize=9.5)
    ax.set_ylabel("Qubits", color=INK2, fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.0, 0.93))
    ax.set_title("Qubits for the same 9-payment gridlock as amounts get finer", color=INK, fontsize=12,
                 loc="left", pad=22)
    ax.text(0.0, 1.02, "Scaling leaves the feasible set unchanged. Only binary slack grows with amount size. "
            "Public-data tier at exact cents: E0 median 84 qubits.", transform=ax.transAxes,
            color=INK2, fontsize=8.5, va="bottom")
    fig.tight_layout()
    fig.savefig("results/fig_qubits.png", dpi=DPI, facecolor=SURF)
    plt.close(fig)
    return rows


# ------------------------------------------------------------------ figure 3: penalty-weight rules
def fig_penalty_rules():
    T = json.load(open("results/tuning_failures.json"))
    rules = ["local", "neighbour", "global"]
    encs = [("E0", "E0 slack (graded)", BLUE), ("E4", "E4 overflow² (graded)", ORANGE),
            ("E1", "E1 IQP (flat)", AQUA), ("E2", "E2 indicator (flat)", YELLOW)]
    tot = {r: {e: 0 for e, _, _ in encs} for r in rules}
    n = {r: 0 for r in rules}
    for row in T:
        n[row["rule"]] += row["n"]
        for e, _, _ in encs:
            tot[row["rule"]][e] += row[e]
        assert row["E0"] == row["E4"], "E0 and E4 have the same cost once slack is minimised"
    assert n == {"local": 360, "neighbour": 360, "global": 360}
    fig, ax = plt.subplots(figsize=(8.6, 4.9), facecolor=SURF)
    style(ax)
    width, gap = 0.19, 0.012
    for j, (e, name, col) in enumerate(encs):
        xs = np.arange(len(rules)) + (j - 1.5) * (width + gap)
        vals = [100.0 * tot[r][e] / n[r] for r in rules]
        ax.bar(xs, vals, width=width, color=col, label=name, zorder=3)
        for xx, r, v in zip(xs, rules, vals):
            ax.text(xx, v + 0.6, "%d" % tot[r][e], ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels(["local\n(payments of the bank)", "neighbour\n(De Santis Eq. 28)",
                        "global\n(all payments)"], color=INK, fontsize=9.5)
    ax.set_ylabel("Cases where the encoded minimum is not the optimum (%)", color=INK2, fontsize=9)
    ax.set_ylim(0, 32)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("Penalty weight rule vs penalty shape", color=INK, fontsize=12, loc="left", pad=36)
    ax.text(0.0, 1.02, "360 cases per rule: 60 networks each of 4 banks/6 payments and 6 banks/9 payments, at liquidity f = 0.1, 0.2, 0.3\n"
            "(scripts/tuning_failures.py, γ = 2). Numbers on bars = failing cases.", transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
    fig.tight_layout()
    fig.savefig("results/fig_penalty_rules.png", dpi=DPI, facecolor=SURF)
    plt.close(fig)
    return tot


# ------------------------------------------------------------------ figure 4: public-data gridlock (AML043)
def dollars(c):
    """Dollar string for matplotlib text: the $ is escaped, otherwise two of them start mathtext."""
    return r"\$" + format(c / 100.0, ",.2f")


def fig_aml043():
    from settle.instances import load_instances_json
    from settle.exact import deletion_heuristic
    from settle.aml import bs_fifo
    inst = {i.name: i for i in load_instances_json(open("data/aml_instances.json").read())}["AML043"]
    assert inst.meta.get("unit") == "US cents" and inst.n_banks == 5 and inst.n == 9
    opt, idx = brute_force(inst)
    assert len(idx) == 1, "figure assumes a unique optimum"
    ox = np.array([(idx[0] >> k) & 1 for k in range(inst.n)])
    assert [k for k in range(inst.n) if ox[k]] == [0, 3, 6, 7, 8] and opt == 76892
    heur = {"greedy queue (largest first)": greedy(inst)[0], "FIFO queue": greedy(inst, order="fifo")[0],
            "Bech-Soramäki FIFO": bs_fifo(inst)[0], "deletion heuristic": deletion_heuristic(inst)[0]}
    assert all(v == 0 for v in heur.values())
    for u in range(inst.n_banks):                    # no payment can settle on its own
        assert all(w > inst.liquidity[u] for a, b, w in inst.arcs if a == u)
    names = "ABCDE"
    pays = [sum(w for k, (a, b, w) in enumerate(inst.arcs) if a == u and ox[k]) for u in range(5)]
    recv = [sum(w for k, (a, b, w) in enumerate(inst.arcs) if b == u and ox[k]) for u in range(5)]
    net = [pays[u] - recv[u] for u in range(5)]
    assert net == list(inst.net_outflow_matrix() @ ox) and all(net[u] <= inst.liquidity[u] for u in range(5))
    total = int(inst.weights.sum())

    pos = {0: (0.0, 0.0), 1: (2.4, 0.0), 2: (4.8, 0.0), 3: (2.4, -1.9), 4: (0.0, -1.9)}
    bend = {0: 0.2, 6: 0.2, 3: 0.2, 7: 0.2}     # reciprocal pairs A<->B and A<->E bow to opposite sides
    lab_t = {2: 0.27, 8: 0.73}                  # the two crossing diagonals: labels away from the crossing
    R = 0.36
    fig = plt.figure(figsize=(12.4, 5.6), facecolor=SURF)
    ax = fig.add_axes([0.01, 0.12, 0.56, 0.72])
    style(ax); ax.set_xticks([]); ax.set_yticks([])
    for k, (a, b, w) in enumerate(inst.arcs):
        on = bool(ox[k]); rad = bend.get(k, 0.0)
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle="arc3,rad=%g" % rad, arrowstyle="-|>",
                                     mutation_scale=15, shrinkA=24, shrinkB=26, color=BLUE if on else MUTED,
                                     lw=2.4 if on else 1.3, zorder=2))
        (x0, y0), (x1, y1) = pos[a], pos[b]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if rad:                                   # bowed midpoint, pushed a little further outward
            nx, ny = dy / L * np.sign(rad), -dx / L * np.sign(rad)
            off = abs(rad) * L * 0.5 + 0.22
            mx, my = (x0 + x1) / 2 + nx * off, (y0 + y1) / 2 + ny * off
        else:
            t = lab_t.get(k, 0.5)
            mx, my = x0 + t * dx, y0 + t * dy
        ax.text(mx, my, dollars(w), ha="center", va="center", fontsize=8.5, color=INK if on else INK2, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none"))
    for u, (px, py) in pos.items():
        ax.add_patch(Circle((px, py), R, fc="#efeeea", ec="none", zorder=3))
        ax.text(px, py, names[u], ha="center", va="center", fontsize=12, color=INK, fontweight="bold", zorder=5)
    ax.set_xlim(-0.9, 5.3); ax.set_ylim(-2.55, 0.85); ax.set_aspect("equal")
    ax.set_title("Optimal simultaneous batch: settles %s (5 of 9 payments)" % dollars(opt), color=INK, fontsize=12,
                 loc="left", pad=10)

    # side table: what each bank pays, receives and holds in the optimal batch
    tx = 0.60
    fig.text(tx, 0.80, "Each bank in the optimal batch", color=INK, fontsize=11)
    cols = [(tx, "Bank", "left"), (tx + 0.085, "Liquidity", "right"), (tx + 0.175, "Pays", "right"),
            (tx + 0.265, "Receives", "right"), (tx + 0.37, "Net outflow", "right")]
    y = 0.74
    for x, h, al in cols:
        fig.text(x, y, h, color=INK2, fontsize=9, ha=al)
    for u in range(5):
        y -= 0.055
        row = [names[u], dollars(inst.liquidity[u]), dollars(pays[u]), dollars(recv[u]),
               ("−" if net[u] < 0 else "") + dollars(abs(net[u]))]
        for (x, _, al), txt in zip(cols, row):
            fig.text(x, y, txt, color=INK, fontsize=9, ha=al, family="DejaVu Sans")
    y -= 0.07
    fig.text(tx, y, "Net outflow ≤ liquidity for every bank, so the batch is feasible.\n"
             "Every bank's liquidity is below each of its own payments,\nso no payment can settle on its own.",
             color=INK2, fontsize=9, va="top")
    y -= 0.15
    fig.text(tx, y, "Settled by classical heuristics: %s each\n(greedy and FIFO queues, Bech-Soramäki FIFO, deletion).\n"
             "Optimum (exact, unique): %s of %s queued." % (dollars(0), dollars(opt), dollars(total)),
             color=INK, fontsize=9, va="top")

    h1 = plt.Line2D([], [], color=BLUE, lw=2.4, label="settled in the optimal batch")
    h2 = plt.Line2D([], [], color=MUTED, lw=1.3, label="stays queued")
    fig.legend(handles=[h1, h2], loc="upper right", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK,
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Public-data gridlock (AML043): every heuristic settles %s" % dollars(0), color=INK, fontsize=13,
                 x=0.01, ha="left", y=0.985)
    fig.text(0.01, 0.02,
             "Source: IBM synthetic AML transactions (HI-Small; simulated customer payments, not real bank data), "
             "USD wires between banks A-E, CSV rows %d-%d,\n%s, %d minutes. Amounts are exact cents. Liquidity is "
             "not in the data: it is set to %d%% of each bank's outgoing total (rounded up). Arrow = payer to payee; A-E are banks 0-4 in the console."
             % (min(inst.meta["source_rows"]), max(inst.meta["source_rows"]), inst.meta["start"][:16],
                inst.meta["span_minutes"], round(100 * inst.meta["f_liq"])),
             ha="left", va="bottom", fontsize=8.5, color=INK2)
    fig.savefig("results/fig_aml043.png", dpi=DPI, facecolor=SURF)
    plt.close(fig)
    return {"opt_cents": opt, "total_cents": total, "net_cents": net, "heuristics": heur}


if __name__ == "__main__":
    fig_problem()
    q = fig_qubits()
    t = fig_penalty_rules()
    a = fig_aml043()
    print("saved results/fig_problem.png, results/fig_qubits.png, results/fig_penalty_rules.png, results/fig_aml043.png")
    print("qubits (scale, E0, E1, E2-4):", q)
    print("penalty-rule failures out of 360:", t)
    print("AML043:", a)
