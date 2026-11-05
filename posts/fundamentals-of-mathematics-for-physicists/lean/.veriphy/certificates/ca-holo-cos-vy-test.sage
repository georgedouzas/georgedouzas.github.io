# name: ca-holo-cos-vy-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.cos_vy
# description: v_y = -sin x cosh y for v = -sin x sinh y at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'exact ((Real.hasDerivAt_sinh _).const_mul _).neg.deriv'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
