# name: ca-holo-log-re-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.log_re_eq
# description: Re(Log z + 2πik) = ln √(x^2 + y^2) at sampled k, x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'simp [Complex.log_re, Complex.norm_def, Complex.normSq_apply, sq]'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
