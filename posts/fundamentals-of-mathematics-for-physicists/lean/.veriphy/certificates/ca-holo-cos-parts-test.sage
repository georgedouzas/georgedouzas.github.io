# name: ca-holo-cos-parts-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_parts
# description: cos(x + iy) = cos x cosh y - i sin x sinh y at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'push_cast\nrw [Complex.cos_add_mul_I]\nring'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
