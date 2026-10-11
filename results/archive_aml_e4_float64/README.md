# E4 on the AML tier: run, then withdrawn

E4 (overflow-squared) was run on the 22 AML batches (44 rows here), then removed from the results.
At cent scale its coefficients reach ~1e26. After the QAOA normalisation (divide by the largest
coefficient), the optimum and the next-best settlement differ by ~1e-16 (median), below float64
resolution, and the float64 minimum of the normalised Hamiltonian is the optimum on only 11 of 22
batches. In exact (rational) arithmetic E4 is exact on 22/22 under every weight rule. So these rows
measure rounding, not the encoding. Kept for transparency.
