# name: ca-holo-cube-uy-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.cube_uy
# description: u_y = -6xy for u = x^3 - 3xy^2 at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'refine ((hasDerivAt_const _ _).sub ((hasDerivAt_pow 2 _).const_mul _)).deriv.trans ?_\nnorm_num\ntry ring'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
