# name: ca-holo-exp-parts-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.exp_parts
# description: e^{x+iy} = e^x cos y + i e^x sin y at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'push_cast\nrw [Complex.exp_add, Complex.exp_mul_I]\nring'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
