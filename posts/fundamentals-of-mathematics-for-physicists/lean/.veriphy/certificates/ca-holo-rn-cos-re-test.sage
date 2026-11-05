# name: ca-holo-rn-cos-re-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.rn_cos_eq_re
# description: r^n cos nθ = Re (r e^{iθ})^n at sampled n, r, θ; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'rw [mul_pow, ← Complex.exp_nat_mul, ← Complex.ofReal_pow, Complex.re_ofReal_mul]\ncongr 1\nrw [← Complex.exp_ofReal_mul_I_re]\ncongr 1\ncongr 1\npush_cast <;> ring'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
