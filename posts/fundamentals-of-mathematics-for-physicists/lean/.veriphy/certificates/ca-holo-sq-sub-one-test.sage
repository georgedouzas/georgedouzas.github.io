# name: ca-holo-sq-sub-one-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_sub_one_eq_zero_iff
# description: z^2 - 1 = 0 iff z = ±1 at z = 0, 1, -1; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'norm_num [Complex.ext_iff, pow_two]', True),
    (['1'], 'norm_num [Complex.ext_iff, pow_two]', False),
    (['(-1)'], 'norm_num [Complex.ext_iff, pow_two]', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
