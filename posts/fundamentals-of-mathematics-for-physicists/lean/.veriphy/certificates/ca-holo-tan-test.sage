# name: ca-holo-tan-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tan_differentiableAt
# description: tan differentiable at z = 0, π, 2π; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '(by simp)'], 'exact Complex.differentiableAt_tan.mpr (by simp)', True),
    (['(Real.pi : ℂ)', '(by simp)'], 'exact Complex.differentiableAt_tan.mpr (by simp)', False),
    (['(2 * (Real.pi : ℂ))', '(by simp)'], 'exact Complex.differentiableAt_tan.mpr (by simp)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
