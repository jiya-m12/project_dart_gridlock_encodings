# Method: problem, encodings, circuit, protocol, metrics

This file defines every quantity used in README.md, RESULTS.md and the result files. Each formula names the function that implements it, so it can be checked against the code. Notation: bits are 0/1, sums run over payments k unless stated, and [·] is 1 when the condition holds and 0 otherwise.

![The gridlock example: RTGS queue vs optimal batch](results/fig_problem.png)

## 1. Problem

**Data.**
- Banks u = 1..B, each with liquidity C_u ≥ 0 (an integer).
- Queued payments k = 1..N. Payment k goes from bank a_k to bank b_k, with integer amount w_k ≥ 1.

**Decision.** x_k = 1 if payment k settles in this batch, and 0 if it stays queued. All selected payments settle at the same instant.

**Net outflow of bank u** (`Instance.net_outflow_matrix`):

  net_u(x) = Σ_{k: a_k = u} w_k x_k − Σ_{k: b_k = u} w_k x_k

**Overdraft of bank u:** g_u(x) = net_u(x) − C_u. Bank u is within its liquidity when g_u(x) ≤ 0.

**Problem:**

  maximise  W(x) = Σ_k w_k x_k
  subject to  g_u(x) ≤ 0 for every bank u
  with  x ∈ {0,1}^N

Properties:
- Settling nothing (x = 0) is always feasible.
- Incoming payments offset outgoing ones, which is why settling together can clear a gridlock that settling one at a time cannot.
- With one bank and only outgoing payments, the problem is subset-sum, so it is NP-hard in general. We make no hardness claim for our small instances.

**Relation to prior work.** This is the bank-clearing / gridlock-resolution problem (Güntzer et al. 1998; Bech & Soramäki 2001).
- It is De Santis et al.'s MPBS with only the lower bound of CAP/FLOOR (Eq. 43 with FL(u) = −C_u) and without the IN/OUT rule.
- It is also the cash-only special case of delivery-versus-payment securities settlement (Braine et al.; Martini et al.; Mayhew): drop the securities constraint and their cash constraint is exactly g_u(x) ≤ 0.

**Ground truth.**
- Brute force over all 2^N selections (`exact.brute_force`).
- MILP with HiGHS (`exact.milp`).
- Optimal values are compared with an absolute tolerance of 1e-6 (`exact.is_value`). Amounts are integers (US cents on the AML tier), so this is an exact comparison; a relative tolerance would count a selection one cent short of a \$10,000 optimum as optimal.

**Binding banks.** A bank can only violate its constraint if its total outgoing amount exceeds its liquidity: Σ_{k: a_k = u} w_k > C_u (`encodings.binding_banks`). Only binding banks get a penalty.

**Degree cap.** Every bank has at most 5 incident payments. So every penalty below is a function of at most 5 payment bits, plus that bank's own slack bits.

## 2. Encodings

Every encoding produces one cost function to **minimise** over the payment bits x and the slack bits s:

  H(x, s) = −Σ_k w_k x_k + Σ_{u binding} λ_u · P_u(x_u, s_u)

Here x_u are the bits of the payments touching bank u, and s_u are slack bits used only by bank u (`encodings.Encoding`, one `Block` per bank).

| | Penalty P_u | Extra qubits per binding bank | Order | Source | Code |
|---|---|---|---|---|---|
| E0 | (g_u + Σ_j 2^j s_{u,j})², I_u = Σ_{k: b_k = u} w_k (inflow) | ⌈log₂(C_u + I_u + 1)⌉, at least 1 | 2 (QUBO) | textbook binary slack | `e0_standard` |
| E1 | −Q_u(x_u, s_u), Q_u quadratic (conditions below) | fewest possible: 0-2 here | 2 (QUBO) | De Santis et al., Sec. III A, Eqs. 10-11 | `e1_iqp`, `iqp.derive_min_slack` |
| E2 | [g_u > 0] | 0 | ≤ 5 (HUBO) | Bucher et al. (IF-QAOA), Sec. IV | `e2_indicator` |
| E3 | −λ₁ h_u + λ₂ h_u², h_u = −g_u = C_u − net_u (no λ_u) | 0 | 2 (QUBO) | Montañez-Barrera et al., Eq. 25 | `e3_unbalanced` |
| E4 | max(0, g_u)² | 0 | ≤ 5 (HUBO) | Bucher et al. Eq. 18; Hess et al. Eq. 11 | `e4_overflow_sq` |

**E1 conditions** (De Santis Eq. 10).
- If g_u(x_u) > 0: Q_u(x_u, s) ≤ −1 for every s.
- If g_u(x_u) ≤ 0: Q_u(x_u, s) ≤ 0 for every s, and = 0 for at least one s.

We search k = 0, 1, 2, … slack bits and stop at the first k for which such a Q_u exists. Among those polynomials we take the one with the smallest largest coefficient (exact MILP, then an LP polish). Q_u depends only on which local assignments are feasible, so polynomials are cached by that pattern. Every shipped polynomial is re-verified, and `scripts/check_iqp_minimal.py` checks that fewer slack bits are impossible.

**E3 weights.**
- λ₁ and λ₂ are tuned on separate training instances (disjoint seeds), never with knowledge of the test optima (`scripts/tune_e3.py`).
- The criterion is the fraction of training instances whose ground state is the optimum. Ties go to the smaller λ₂, then the smaller λ₁.
- Chosen values: main set (1.5, 0.2), T0 (1.5, 0.75), C (0, 0.3). On C, λ₁ = 0, so E3 reduces to a plain squared penalty there.
- The weights are in amount units and do not transfer to cents, so E3 is not run on the AML tier.

**Penalty weights λ_u** (E0, E1, E2, E4; `encodings.penalty_weights`, γ = 2):
- **local:** λ_u = γ · (total amount of the payments touching u);
- **neighbour:** λ_u = γ · (total amount of the payments touching u or any bank that shares a payment with u), each payment counted once. This is De Santis Eq. 28 for a diagonal objective. Their MPBS form, Eq. 68, counts a payment between two neighbours twice and so gives larger weights;
- **global:** λ_u = γ · Σ_k w_k for every bank.

**Why the encodings are exact (E0, E1, E2, E4).** Let p_u(x_u) = min over s_u of P_u(x_u, s_u), the penalty after the best slack choice. With integer amounts, p_u = 0 when g_u ≤ 0, and p_u ≥ 1 when g_u > 0:
- **E2:** by definition.
- **E4:** max(0, g)² ≥ 1 for an integer g ≥ 1.
- **E1:** by the conditions above.
- **E0:** for g_u ≤ 0, the slack value −g_u ≤ C_u + I_u is representable, giving 0. For g_u ≥ 1 every slack value gives (g_u + s)² ≥ g_u² ≥ 1.

So with the global rule and γ > 1, any infeasible x costs at least −W(x) + γ Σ_k w_k > 0 ≥ −W(x*). Every minimiser of H is therefore an optimal settlement. The neighbour and local rules carry no such guarantee:
- The neighbour rule fails on 2 of 360 cases in the tuning study (figure below), for the flat penalties E1/E2, on one payment network (seed 50004, 6 banks/9 payments, f = 0.1 and 0.2). Whether De Santis's Eq. 68 weights would avoid it depends on how neigh(u) is read: with u included they are larger and exact on that network; with u excluded they fail there too.
- Every encoding we run is exact under the neighbour rule on every benchmark instance, except E3, which is approximate by design. This is checked by enumeration (`verify.check_exact`).
- On the AML tier, E2 and E4 are also checked straight from the penalty definitions in Python integers (`scripts/aml_encoding_gates.py`): both are exact on 22/22. E4's float64 polynomial is not (21/22 with neighbour weights), one of the reasons E4 is not run there (Section 3). E0 is too large to enumerate on the AML tier.

![Penalty weight rule vs penalty shape](results/fig_penalty_rules.png)

**E0 and E4 have the same cost once slack is minimised out.** By the argument above, min over s of (g + s)² = max(0, g)². So E0 and E4 have identical reduced cost on every selection x; the tuning study confirms it (equal failure counts in every cell). Comparing E0 with E4 therefore isolates the price of the slack qubits.

**Qubits.**
- E0: N + Σ_u ⌈log₂(C_u + I_u + 1)⌉.
- E1: N plus 0-2 slack bits per binding bank.
- E2, E3, E4: exactly N.

Scaling every amount by s leaves the feasible set unchanged. Only E0 grows, by about log₂ s bits per binding bank (figure below).

![Qubits vs amount resolution](results/fig_qubits.png)

## 3. From penalty to circuit

**Higher-order expansion (E2, E4; `poly.from_table`).** Each penalty f(x_u) is evaluated on all 2^d local assignments (d ≤ 5). These values are turned into the unique multilinear polynomial by Möbius inversion:

  f(x) = Σ_{S ⊆ bank's payments} c_S Π_{i ∈ S} x_i,  c_S = Σ_{T ⊆ S} (−1)^{|S|−|T|} f(1_T)

where 1_T is the assignment with exactly the bits in T set. No approximation and no ancilla is involved.

**Ising form (`poly.to_ising`).** Substitute x_i = (1 − Z_i)/2. Qubit state |1⟩ means payment settled; qubit j is bit j (Qiskit little-endian); slack qubits come after the N payment qubits. The result is H = const + Σ_T h_T Π_{i ∈ T} Z_i.

**Gates (`qaoa.append_cost_layer`).**
- A 1-body term is RZ(2γ h_T).
- A k-body term is a CNOT ladder onto its last qubit, then RZ(2γ h_T), then the ladder undone: 2(k − 1) CNOTs.
- The CNOT counts reported are this ladder count, with no cancellation between terms.

**Normalisation.** The constant is dropped and every coefficient is divided by the largest |h_T| (`qaoa.Problem`). The same rule is used for every encoding.

**Objective visibility.** After normalisation, what matters is how large the settlement-value part of H is relative to the largest coefficient.
- We report max |objective coefficient| ÷ max |coefficient| (`scripts/analyse.py`, `visibility`), median on the main set: E0 0.00045, E1 0.12, E2 0.23, E3 0.42, E4 0.0044. So the settlement value is about 50× less visible in E4 than in E2, and comparing E2 with E4 mixes penalty shape with this scale effect. The values for every set are printed under the result tables in RESULTS.md.
- E0 and E4 penalties grow with the square of the overdraft. At exact cents (AML tier) E4's coefficients reach 7.7e26, and after normalisation the optimum and the next-best settlement differ by about 1e-16, below float64 resolution (2.2e-16). The float64 minimum of the normalised Hamiltonian is the optimum on only 11 of 22 batches. **E4 is therefore not run on the AML tier**; the withdrawn rows are in `results/archive_aml_e4_float64/`.
- E1 and E2 grow linearly with amounts and stay usable at cents (visibility 0.33 and 0.44 on the AML tier).

## 4. QAOA protocol (`qaoa.optimise_sweep`, `scripts/run_benchmark.py`)

**State:**

  |ψ(γ, β)⟩ = Π_{l = p..1} e^{−iβ_l Σ_j X_j} e^{−iγ_l Ĥ} |+⟩^{⊗n}

Here Ĥ is the normalised cost and e^{−iβX} = RX(2β). It is simulated exactly as a numpy statevector; the Qiskit circuit is checked to match it to 1e-9.

**Objective.**
- Primary: CVaR_α with α = 0.25, the mean energy of the lowest-energy 25% of the output distribution (Barkoutsos et al. 2020), computed exactly from the probabilities: CVaR_α = (1/α) ∫₀^α F⁻¹(q) dq, where F is the energy distribution function.
- Secondary: ⟨Ĥ⟩.
- The objective was chosen on training instances before the benchmark (`results/pilot_objective.json`).

**Optimiser.** COBYLA, at most 1000 iterations per start, initial step 0.3.
- p = 1: a linear-ramp start plus 2 random starts: γ uniform in (0, π), β uniform in (−π/2, 0).
- p = 2, 3: the previous optimum with a zero layer appended (it starts at the previous optimum's value), its INTERP extension (Zhou et al. 2020), and 2 random starts.
- The best value over starts is kept.
- The start sign of β is negative (`qaoa.BETA_START_SIGN`). Two caveats, both from a 64 × 64 grid over the p = 1 angles on the main set (`scripts/check_p1_grid.py`, `results/p1_grid_check.json`):
  - The best p = 1 point lies at β < 0 on 34, 34 and 36 of 40 instances for E1, E2 and E4, not on all of them.
  - The optimiser is local. Its p = 1 result is more than 0.1% worse than the best grid point on 3 (E1), 10 (E2) and 8 (E4) of 40 instances.

**Control.** An unentangled product state ⊗_j RY(θ_j)|0⟩, so P(bit j = 1) = sin²(θ_j / 2). It uses the same objective, optimiser and budget, with starts at θ_j = π/2 plus 2 random starts (`qaoa.optimise_product`).
- It has n free angles against QAOA's 2p, so it is an unentangled per-qubit baseline, not "QAOA minus entanglement".
- It is the same idea as Martini et al.'s Q-INSP.

## 5. Metrics (`metrics.quality`)

Slack qubits are summed out first: p(x) = Σ_s |⟨x, s|ψ⟩|².
- **P(opt)** = Σ_{x optimal} p(x): the probability that one sample is an optimal settlement.
- **P(feasible)** = Σ_{x feasible} p(x).
- **× random** (enrichment) = P(opt) ÷ (number of optimal x / 2^N), compared with uniform random guessing over payment selections.
- **Expected-value ratio** = Σ_x p(x) · W(x) · [x feasible] ÷ W(x*). Infeasible samples count as 0.
- **Shots to first optimum** = 1 / P(opt), the mean of a geometric distribution. On a backend, the sampled frequency is used.
- **Statistics:**
  - Medians over instances, with 95% bootstrap confidence intervals (2000 resamples).
  - Paired Wilcoxon signed-rank tests between encodings on the same instances, ties dropped (only when at least 5 pairs differ).
  - Holm-Bonferroni correction over all encoding pairs within a set; RESULTS.md reports both the raw and the adjusted p.

**CVaR ceiling.** CVaR_0.25 stops improving once about 25% of the probability sits on the lowest-energy state. So P(opt) near 0.25 under CVaR marks the objective's limit, not the encoding's. We count runs that reached it with the proxy P(opt) ≥ 0.95α (for E0/E1 the slack must also be right; for E3 the minimum need not be the optimum).

## 6. Classical baselines (`exact.greedy`, `aml.bs_fifo`, `exact.deletion_heuristic`)

- **RTGS queue (greedy):** go through the queue largest payment first and settle a payment if the payer's current balance covers it. Repeat passes until nothing changes.
- **Bech-Soramäki FIFO:** try the whole queue. While some banks are short, each short bank removes its most recently queued selected outgoing payment. Then add payments back in queue order.
- **Deletion heuristic:** start from everything. Repeatedly drop, at the bank with the largest shortfall, the smallest selected outgoing payment that covers it (or the largest if none does). Then add dropped payments back, largest first, wherever feasibility allows.
- **Exact:** brute force and MILP.

## 7. Instance sets

- **Main set:** 40 instances, 4-5 banks, 6-8 payments, amounts 1-10.
- **T0:** 20 instances, amounts 1-3, so E0 fits in a simulator.
- **C:** 20 instances, 6 banks, 10 payments.

All three are a directed ring plus random chords, with C_u = ⌈f · outgoing_u⌉ for f ∈ {0.2, 0.3, 0.4}. An instance is kept only if the optimum beats the greedy queue (`scripts/make_instances.py`, fixed seeds).

**AML tier:** 22 hard batches of USD wire payments from the IBM synthetic AML dataset, in exact cents, with the same liquidity rule. See DATA.md.
