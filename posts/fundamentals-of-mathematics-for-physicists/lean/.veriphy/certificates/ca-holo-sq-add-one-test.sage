# name: ca-holo-sq-add-one-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.sq_add_one_eq_zero_iff
# description: z^2 + 1 = 0 iff z = ±i, at z = 0, i, -i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'norm_num [Complex.ext_iff, pow_two]', True),
    (['Complex.I'], 'norm_num [Complex.ext_iff, pow_two]', False),
    (['(-Complex.I)'], 'norm_num [Complex.ext_iff, pow_two]', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
