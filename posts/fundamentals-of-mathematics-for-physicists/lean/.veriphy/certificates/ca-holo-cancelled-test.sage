# name: ca-holo-cancelled-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cancelled_differentiableAt
# description: 1/(z^2 - 1) differentiable at z = 0, i, 2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '(by norm_num [Complex.ext_iff])', '(by norm_num [Complex.ext_iff])'], 'exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff, pow_two])', True),
    (['Complex.I', '(by norm_num [Complex.ext_iff])', '(by norm_num [Complex.ext_iff])'], 'exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff, pow_two])', False),
    (['2', '(by norm_num [Complex.ext_iff])', '(by norm_num [Complex.ext_iff])'], 'exact DifferentiableAt.div (by fun_prop) (by fun_prop) (by norm_num [Complex.ext_iff, pow_two])', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
