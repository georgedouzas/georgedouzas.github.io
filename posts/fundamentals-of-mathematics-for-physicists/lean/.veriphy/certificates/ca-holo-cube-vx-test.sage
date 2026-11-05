# name: ca-holo-cube-vx-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_vx
# description: v_x = 6xy for v = 3x^2 y - y^3 at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'refine ((((hasDerivAt_pow 2 _).const_mul 3).mul_const _).sub_const _).deriv.trans ?_\nnorm_num\ntry ring'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
