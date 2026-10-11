"""Settlement instances: banks, queued payments (arcs) and liquidity.

Problem (liquidity-only multilateral settlement / RTGS gridlock resolution):
    maximise   sum_k w_k x_k
    subject to for every bank u:
               sum_{k: a_k=u} w_k x_k - sum_{k: b_k=u} w_k x_k <= C_u
x_k = 1 means payment k (a_k -> b_k, amount w_k) settles in this batch.
Incoming settled payments offset outgoing ones (simultaneous settlement).
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Instance:
    name: str
    n_banks: int
    arcs: list          # list of (debtor, creditor, amount), amounts are positive ints
    liquidity: list     # C_u >= 0 per bank
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.arcs)

    @property
    def weights(self):
        return np.array([a[2] for a in self.arcs], dtype=float)

    def out_arcs(self, u):
        return [k for k, (a, b, w) in enumerate(self.arcs) if a == u]

    def in_arcs(self, u):
        return [k for k, (a, b, w) in enumerate(self.arcs) if b == u]

    def incident(self, u):
        """Sorted indices of payments touching bank u (the variables of u's constraint)."""
        return sorted(set(self.out_arcs(u)) | set(self.in_arcs(u)))

    def degree(self, u):
        return len(self.incident(u))

    def local_coeffs(self, u):
        """(vars, coeffs): net outflow of u = sum coeffs[i] * x[vars[i]]."""
        vs = self.incident(u)
        cs = []
        for k in vs:
            a, b, w = self.arcs[k]
            cs.append((w if a == u else 0) - (w if b == u else 0))
        return vs, cs

    def net_outflow_matrix(self):
        """A (n_banks x n) with A[u,k] = +w if u pays k, -w if u receives k."""
        A = np.zeros((self.n_banks, self.n))
        for k, (a, b, w) in enumerate(self.arcs):
            A[a, k] += w
            A[b, k] -= w
        return A

    def is_feasible(self, x):
        x = np.asarray(x, dtype=float)
        return bool(np.all(self.net_outflow_matrix() @ x <= np.asarray(self.liquidity) + 1e-9))

    def value(self, x):
        return float(self.weights @ np.asarray(x, dtype=float))

    def to_dict(self):
        return {"name": self.name, "n_banks": self.n_banks,
                "arcs": [list(map(int, a)) for a in self.arcs],
                "liquidity": [int(c) for c in self.liquidity], "meta": self.meta}

    @staticmethod
    def from_dict(d):
        return Instance(d["name"], d["n_banks"], [tuple(a) for a in d["arcs"]],
                        list(d["liquidity"]), d.get("meta", {}))


def save_instances(instances, path):
    with open(path, "w") as f:
        json.dump([i.to_dict() for i in instances], f, indent=1)


def load_instances_json(text):
    return [Instance.from_dict(d) for d in json.loads(text)]


def gridlock_instance():
    """6 banks, 9 payments, two interlocking cycles with cross-links.
    Same data as the disclosed pre-window notebook (dataset_gridlock.csv)."""
    arcs = [(0, 1, 8), (1, 2, 7), (2, 0, 9), (2, 3, 6), (3, 4, 5),
            (4, 5, 7), (5, 2, 8), (1, 3, 4), (4, 0, 3)]
    return Instance("gridlock6x9", 6, arcs, [2, 3, 4, 2, 3, 2],
                    {"source": "hand-built scenario from disclosed prior notebook"})


def generate(n_banks, n_arcs, seed, f_liq=0.3, w_max=10, max_degree=5, max_tries=1000):
    """Random settlement batch with cycles: a directed ring over all banks plus
    random chords, no self-loops, every bank with at most `max_degree` incident
    payments. Liquidity C_u = ceil(f_liq * total outgoing of u).
    Returns None if the degree cap cannot be met."""
    rng = np.random.default_rng(seed)
    if n_arcs < n_banks:
        raise ValueError("need n_arcs >= n_banks for the ring")
    for _ in range(max_tries):
        perm = rng.permutation(n_banks)
        pairs = [(int(perm[i]), int(perm[(i + 1) % n_banks])) for i in range(n_banks)]
        deg = np.zeros(n_banks, int)
        for a, b in pairs:
            deg[a] += 1
            deg[b] += 1
        ok = True
        while len(pairs) < n_arcs:
            cand = [(a, b) for a in range(n_banks) for b in range(n_banks)
                    if a != b and deg[a] < max_degree and deg[b] < max_degree]
            if not cand:
                ok = False
                break
            a, b = cand[rng.integers(len(cand))]
            pairs.append((a, b))
            deg[a] += 1
            deg[b] += 1
        if not ok:
            continue
        ws = rng.integers(1, w_max + 1, size=len(pairs))
        arcs = [(a, b, int(w)) for (a, b), w in zip(pairs, ws)]
        out = np.zeros(n_banks)
        for a, b, w in arcs:
            out[a] += w
        liq = [int(math.ceil(f_liq * o)) for o in out]
        return Instance(f"rand_b{n_banks}_a{n_arcs}_f{f_liq}_s{seed}", n_banks, arcs, liq,
                        {"generator": "ring+chords", "seed": seed, "f_liq": f_liq,
                         "w_max": w_max, "max_degree": max_degree})
    return None
