# Data

## Two tiers, reported separately
- **Controlled tier** (`data/bench_instances.json`, frozen): random ring + chords, amounts 1..10, liquidity f in {0.2, 0.3, 0.4}. The ring and the uniform small amounts are modelling assumptions.
- **Public-data tier** (`data/aml_instances.json`, pool in `data/aml_pool.json`): settlement batches derived from a public dataset, described below.

## What the public-data tier is
Real interbank settlement (RTGS) data is confidential and not public. We derive settlement batches from the public, peer-reviewed **IBM synthetic AML transaction dataset**: Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023 Datasets & Benchmarks (arXiv:2306.16424), file HI-Small_Trans.csv from Kaggle. That dataset is itself simulated by an agent-based generator and describes customer payments, not settlement obligations between banks. **Liquidity is not in the data**; it is assigned by the same stated rule as the controlled tier.
Nothing here is real bank data or RTGS data.

Licence: the source is CDLA-Sharing-1.0. The two derived JSON files are published under CDLA-Sharing-1.0 as modified data (see `data/AML_DATA_LICENSE.txt`). Code is MIT.

## What is in the repository and what is not
- **Not included:** the 475 MB source CSV (HI-Small_Trans.csv). It exceeds the 2 MB per-file limit, and it is freely downloadable: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- **Included:** every settlement batch we derived (`data/aml_pool.json`, 548 batches, 476 KB), the 22-instance QAOA set (`data/aml_instances.json`), and the scripts that produced them. The project runs entirely from these files and never needs the CSV.
- **Traceable:** each instance stores the row number of every payment in the original CSV (`meta.source_rows`), the exact amounts in cents, the timestamp of the first payment, and the liquidity level used. Anyone with the Kaggle file can check any instance line by line.
- **Verifiable in three commands:** download the CSV (SHA-256 `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`), then run `data/aml_profile_filter.py`, `data/gen_aml_tier.py` and `data/validate_aml_tier.py`. The generator reproduces the shipped files byte for byte, and the validator rebuilds and re-solves all 548 instances independently (0 failures).

## One batch, end to end (AML043)
Nine USD wire payments among five banks, 18:56-19:44 on 2022-09-01 (CSV rows 930679 ... 968369). Bank IDs are renumbered A-E.

| # | From → To | Amount |
|---|---|---|
| 0 | A → B | \$118.30 |
| 1 | C → B | \$585,832.01 |
| 2 | A → D | \$191.73 |
| 3 | E → A | \$4.95 |
| 4 | B → D | \$1,078.23 |
| 5 | D → E | \$1,293.85 |
| 6 | B → A | \$405.09 |
| 7 | A → E | \$224.85 |
| 8 | E → B | \$15.73 |

Liquidity (20% of each bank's outgoing total): A \$106.98, B \$296.67, C \$117,166.41, D \$258.77, E \$4.14. Every bank's liquidity is smaller than each of its own outgoing payments, so no payment can settle on its own. Greedy, FIFO, deletion and Bech-Soramäki FIFO all settle \$0: the batch is gridlocked. Settling payments 0, 3, 6, 7 and 8 together is feasible, because incoming payments cover outgoing ones. That clears \$768.92, which is the unique optimum.

## Pipeline (reproducible)
1. `data/aml_profile_filter.py` (run locally on the 475 MB CSV, standard library): keeps rows with From Bank != To Bank, US Dollar paid and received, Wire or ACH. Output `aml_usd_interbank.csv.gz` (264,311 rows) and `aml_profile.json`. Source SHA-256 `b19d39f5...5c040`, 5,078,345 rows.
2. `python3 data/gen_aml_tier.py aml_usd_interbank.csv.gz` builds the tier (deterministic; re-runs give identical files).
3. `python3 data/validate_aml_tier.py aml_usd_interbank.csv.gz` rebuilds every instance from its source rows and re-solves it with its own integer enumeration (no shared solver code). Result: 548 instances, 0 failures. It was also checked to fail on deliberately corrupted files.

## Batch rule (frozen in `data/gen_aml_tier.py`)
- Only **Wire** payments (64,244 rows). Wire is the RTGS analogue; ACH is deferred net settlement and is not used. None of the USD inter-bank Wire rows is labelled laundering (all 1,654 labelled rows in the filtered file are ACH), so the injected laundering cycles cannot create the gridlock we select for.
- 12-hour epochs. In each epoch, disjoint groups of 5 banks are built from payment **counts** only (seed with the pair that trades most, add the bank with most payments to the group).
- A batch = consecutive payments (in submission order) inside one group and epoch, closed before a bank would exceed 5 payments or the batch 9 payments; batches under 6 payments are discarded. Each source row is used at most once.
- Amounts: **exact integer cents**, no rounding. Liquidity C_u = ceil(f · outgoing of u), exact rational arithmetic, one f per batch drawn from {0.2, 0.3, 0.4} with a fixed seed.
- QAOA set (`aml_instances.json`): batches that are not trivially settleable (some bank's liquidity is below its net outflow, the Bech & Soramäki lower bound) and where the optimum beats value-ordered greedy. The controlled tiers use only the second condition (optimum beats greedy), so 12 of their 80 instances can settle in full; the AML tier also excludes those.
- How the parameters were chosen: we compared a few rule variants (random vs traffic grouping, core size, group size, epoch length) on batch counts and classical hardness counts only. No QAOA was run on this tier before freezing.

## What the data shows (all classical, exact)
| | Count |
|---|---|
| Batches built | 548 (4,181 payments; 55,429 Wire rows cross groups, 4,634 in short batches) |
| Contain a directed payment cycle | 404 (74%) |
| Value-ordered greedy already optimal | 525 (96%) |
| **Hard (QAOA set)** | **22** (sizes 4-5 banks, 6-9 payments) |
| Hard where the deletion heuristic is still optimal | 17/22 |
| Hard where Bech-Soramäki FIFO is still optimal | 12/22 |
| Hard where the optimum beats all four classical heuristics | 2/22 |

Liquidity sweep over all 548 batches (hard = optimum beats greedy):

| f | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|
| hard | 19 | 26 | 28 | 24 |
| optimum beats every heuristic | 1 | 6 | 4 | 2 |

**Rounding amounts changes the problem.** Inside a batch, the largest payment is a median 21,000 times the smallest (90th percentile 1.1 million). After rounding to integers 1..Q (largest = Q) and recomputing liquidity, how often an optimal set of the rounded instance is still optimal for the exact one:

| Q | 10 | 100 | 1,000 | 10,000 | 10^6 |
|---|---|---|---|---|---|
| optimum preserved, all 548 | 65 | 170 | 302 | 423 | 540 |
| optimum preserved, 22 hard | 5 | 10 | 15 | 16 | 22 |
| "hard" after rounding | 72 | 65 | 39 | 29 | 24 |

Coarse rounding both changes the answer and manufactures difficulty (72 hard at Q=10 against 22 exact).

Orientation only, not validation: the mean Wire payment is 680× the median (Fedwire 2004: \$3M vs \$30K, about 100×; Soramäki et al., Physica A 379, 2007). Mean within-batch reciprocity is 0.23 (Fedwire network-wide: 0.22). Batch-level and network-level reciprocity are different quantities.

## Encodings on exact amounts (`scripts/aml_encoding_gates.py`, 22 hard instances)
| | E0 slack | E1 IQP | E2 indicator | E4 overflow² |
|---|---|---|---|---|
| Exact, local / neighbour / global | not checked (84 qubits median) | 21 / 22 / 22 | 21 / 22 / 22 | 22/22 by definition; polynomial 20 / 21 / 21 |
| Qubits, median (max) | 84 (122) | 8 (12) | 8 (9) | 8 (9) |
| Coefficient range max/min, median (neighbour) | n/a | 4.2e3 | 2.4e3 | 3.5e6 (max 1.5e15) |
| For comparison, main set (neighbour) | qubits 24 (median) | | range 21 (median) | range 54 (median) |

Penalties that grow with the square of amounts (E0, E3, E4) blow up on realistic amounts. E4's polynomial coefficients reach 7.7e26 in cents. After QAOA's normalisation (divide by the largest coefficient), the optimum and the next-best settlement differ by about 1e-16, below float64 resolution. The float64 minimum of the normalised Hamiltonian is the optimum on only 11 of 22 batches, although E4 is exact on 22/22 in exact arithmetic. E4 is therefore not run on this tier (rows withdrawn to results/archive_aml_e4_float64/). E3's trained weights are in controlled-tier units and do not transfer to cents. E1 and E2 scale linearly and stay usable.

## Limitations
- Batches span hours (median 6 h) because each simulated group of banks trades slowly. A real RTGS queue is minutes.
- 22 hard instances, against 40 in the controlled tier. Under our liquidity rule and grouping, hardness is rare in this data; 55,429 of the 64,244 wire payments cross groups and are not used. That rarity is a result, not a defect, but it depends on these choices.
- One dataset, one liquidity rule, one grouping rule.
