# name: ca-holo-ray-power-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.ray_power
# description: (t e^{iα})^n = t^n e^{inα} at sampled n, t, α; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'rw [mul_pow, ← Complex.exp_nat_mul]\npush_cast\nring_nf'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
