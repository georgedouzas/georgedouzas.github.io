# name: ca-holo-cube-vy-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vy
# description: v_y = 3x^2 - 3y^2 for v = 3x^2 y - y^3 at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = "refine (((hasDerivAt_id' _).const_mul _).sub (hasDerivAt_pow 3 _)).deriv.trans ?_\nnorm_num\ntry ring"
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
