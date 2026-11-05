# name: ca-holo-cos-uy-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_uy
# description: u_y = cos x sinh y for u = cos x cosh y at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'exact ((Real.hasDerivAt_cosh _).const_mul _).deriv'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
