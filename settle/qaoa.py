"""QAOA: exact numpy statevector engine + Qiskit circuit builder.

Conventions (shared by both): qubit j <-> bit j of the basis-state index
(little-endian, Qiskit), payments are qubits 0..n-1, slack after them.
Cost Hamiltonian H = sum_T c_T Z_T (from the binary cost via x = (1-Z)/2),
divided by max |c_T| over non-constant terms (one normalisation rule for all
encodings). Layer: exp(-i gamma H) then exp(-i beta sum_j X_j) = prod RX(2 beta).
H is minimised, so good angles usually have gamma > 0, beta < 0 (see BETA_START_SIGN).
"""
import numpy as np

from . import poly as P


class Problem:
    """Normalised Ising Hamiltonian of an encoding, ready for QAOA."""

    def __init__(self, enc):
        self.enc = enc
        self.n = enc.n_total
        self.n_logical = enc.n_logical
        z = P.to_ising(enc.poly)
        self.const = z.pop((), 0.0)
        self.scale = max(abs(c) for c in z.values())
        self.zterms = {t: c / self.scale for t, c in z.items()}
        self.diag = None
        self.order = None

    def build_diag(self):
        """Diagonal of the normalised H (without constant) over 2^n states."""
        if self.diag is None:
            idx = np.arange(2 ** self.n, dtype=np.int64)
            d = np.zeros(2 ** self.n)
            for t, c in self.zterms.items():
                par = np.zeros_like(idx)
                for q in t:
                    par ^= (idx >> q) & 1
                d += c * (1 - 2 * par)
            self.diag = d
        return self.diag


def apply_mixer(psi, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        v = psi.reshape(-1, 2, 2 ** q)
        a = v[:, 0, :].copy()
        b = v[:, 1, :].copy()
        v[:, 0, :] = c * a + s * b
        v[:, 1, :] = s * a + c * b
    return psi


def statevector(prob, gammas, betas):
    d = prob.build_diag()
    psi = np.full(2 ** prob.n, 2 ** (-prob.n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * d)
        psi = apply_mixer(psi, b, prob.n)
    return psi


def cvar_from_probs(prob, pr, alpha):
    """CVaR_alpha (Barkoutsos et al., Quantum 4, 256 (2020)): mean energy of the
    lowest-energy alpha fraction of the output distribution (exact, from probs)."""
    if prob.order is None:
        prob.order = np.argsort(prob.build_diag(), kind="stable")
    d = prob.diag[prob.order]
    p = pr[prob.order]
    cum = np.cumsum(p)
    full = cum <= alpha
    val = float(np.dot(p[full], d[full]))
    used = float(cum[full][-1]) if full.any() else 0.0
    nxt = int(full.sum())
    if used < alpha and nxt < len(d):
        val += (alpha - used) * d[nxt]
    return val / alpha


def energy(prob, params, objective="mean", alpha=0.1):
    """objective='mean': <H>.  objective='cvar': CVaR_alpha of H."""
    p = len(params) // 2
    psi = statevector(prob, params[:p], params[p:])
    if objective == "mean":
        return float(np.real(np.vdot(psi, prob.build_diag() * psi)))
    return cvar_from_probs(prob, np.abs(psi) ** 2, alpha)


def logical_distribution(prob, psi):
    """Probability of each payment selection x (slack marginalised out)."""
    pr = np.abs(psi) ** 2
    return pr.reshape(-1, 2 ** prob.n_logical).sum(axis=0)


# ------------------------------------------------------------------ Qiskit
def append_cost_layer(qc, zterms, gamma, qubits):
    """exp(-i gamma sum_T c_T Z_T) on the given physical qubits.
    k-body term: CNOT ladder onto the last qubit, RZ(2 gamma c), uncompute."""
    for t, c in sorted(zterms.items(), key=lambda kv: (len(kv[0]), kv[0])):
        qs = [qubits[i] for i in t]
        if len(qs) == 1:
            qc.rz(2 * gamma * c, qs[0])
            continue
        for a, b in zip(qs[:-1], qs[1:]):
            qc.cx(a, b)
        qc.rz(2 * gamma * c, qs[-1])
        for a, b in reversed(list(zip(qs[:-1], qs[1:]))):
            qc.cx(a, b)


def append_qaoa(qc, prob, gammas, betas, qubits=None):
    if qubits is None:
        qubits = list(range(prob.n))
    qc.h(qubits)
    for g, b in zip(gammas, betas):
        append_cost_layer(qc, prob.zterms, g, qubits)
        qc.rx(2 * b, qubits)
    return qc


def qaoa_circuit(prob, gammas, betas, measure=True):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(prob.n)
    append_qaoa(qc, prob, gammas, betas)
    if measure:
        qc.measure_all()
    return qc


def _interp(x):
    """Zhou et al. INTERP: p-layer schedule -> (p+1)-layer schedule."""
    p = len(x)
    xp = np.concatenate([[0.0], x, [0.0]])
    return np.array([(i / p) * xp[i] + ((p - i) / p) * xp[i + 1] for i in range(p + 1)])


# Sign of the mixer angle in the start points. With |+> as the start state, the layer
# exp(-i gamma H) exp(-i beta sum X), and H MINIMISED, the adiabatic-like quadrant is
# gamma > 0, beta < 0 (usually best: p=1 grid optimum at beta < 0 on 34-36/40 main-set instances) (one-qubit check: H = Z, gamma = beta = 0.4 gives P(ground) 0.24
# with beta > 0 and 0.76 with beta < 0). Up to v6 the starts used beta > 0; v7 fixes this
# (tests/test_qaoa_extras.py::test_start_quadrant).
BETA_START_SIGN = -1.0


def optimise_sweep(prob, p_max, restarts=3, seed=0, maxiter=1000, objective="mean", alpha=0.1):
    """Optimise p = 1..p_max. p=1: linear-ramp start (gamma rising, |beta| falling) +
    random starts. Each p>1 starts from the previous optimum with a zero layer appended
    (so the objective can only improve), from its INTERP extension, and from random
    starts. Mixer start angles carry BETA_START_SIGN. Same rule for every encoding.
    Returns {p: (objective value, gammas, betas, n_evals, convergence info)}."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    out = {}
    prev = None
    for p in range(1, p_max + 1):
        starts = []
        if prev is None:
            ramp = (np.arange(p) + 0.5) / p
            starts.append(np.concatenate([0.8 * ramp, BETA_START_SIGN * 0.8 * (1 - ramp)]))
        else:
            g, b = prev
            starts.append(np.concatenate([g, [0.0], b, [0.0]]))
            starts.append(np.concatenate([_interp(g), _interp(b)]))
        while len(starts) < restarts + (0 if prev is None else 1):
            starts.append(np.concatenate([rng.uniform(0, np.pi, p),
                                          BETA_START_SIGN * rng.uniform(0, np.pi / 2, p)]))
        best, evals, hit = None, 0, 0
        for x0 in starts:
            r = minimize(lambda x: energy(prob, x, objective, alpha), x0, method="COBYLA",
                         options={"maxiter": maxiter, "rhobeg": 0.3})
            evals += r.nfev
            hit += int(r.nfev >= maxiter)          # start stopped by the budget, not by convergence
            if best is None or r.fun < best[0]:
                best = (float(r.fun), r.x[:p].copy(), r.x[p:].copy())
        out[p] = best + (evals, {"starts": len(starts), "hit_maxiter": hit})
        prev = (best[1], best[2])
    return out


# ------------------------------------------------------------------ baseline: product state
def product_probs(thetas):
    """Distribution of the unentangled state prod_j RY(theta_j)|0>: P(bit j = 1) = sin^2(theta_j/2)."""
    pr = np.ones(1)
    for t in thetas:                      # qubit j is bit j -> build from qubit 0 upwards
        q1 = np.sin(t / 2) ** 2
        pr = np.concatenate([pr * (1 - q1), pr * q1])
    return pr


def optimise_product(prob, restarts=3, seed=0, maxiter=1000, objective="cvar", alpha=0.25):
    """Same objective and optimiser as QAOA, but no entanglement (Martini et al.'s
    Q-INSP-style control): one RY angle per qubit. Starts: all pi/2 (uniform) + random."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    d = prob.build_diag()

    def f(th):
        pr = product_probs(th)
        return float(pr @ d) if objective == "mean" else cvar_from_probs(prob, pr, alpha)

    starts = [np.full(prob.n, np.pi / 2)] + [rng.uniform(0, np.pi, prob.n) for _ in range(restarts - 1)]
    best, evals, hit = None, 0, 0
    for x0 in starts:
        r = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
        evals += r.nfev
        hit += int(r.nfev >= maxiter)
        if best is None or r.fun < best[0]:
            best = (float(r.fun), r.x.copy())
    return best + (evals, {"starts": len(starts), "hit_maxiter": hit})


# ------------------------------------------------------------------ IonQ: several QAOA blocks, ONE circuit
def packed_circuit(blocks):
    """blocks: list of (Problem, gammas, betas). Each QAOA runs on its own disjoint
    qubit range of a single circuit, so one IonQ job (one circuit) measures all of
    them. Returns (circuit, offsets)."""
    from qiskit import QuantumCircuit
    total = sum(pr.n for pr, _, _ in blocks)
    qc = QuantumCircuit(total)
    offsets, off = [], 0
    for pr, g, b in blocks:
        append_qaoa(qc, pr, g, b, qubits=list(range(off, off + pr.n)))
        offsets.append(off)
        off += pr.n
    qc.measure_all()
    return qc, offsets


def split_counts(counts, blocks, offsets):
    """Per block: probability of each payment selection (slack marginalised)."""
    out = []
    for (pr, _, _), off in zip(blocks, offsets):
        px = np.zeros(2 ** pr.n_logical)
        for key, c in counts.items():
            v = int(key.replace(" ", ""), 2)
            px[(v >> off) & (2 ** pr.n_logical - 1)] += c
        out.append(px / px.sum())
    return out
