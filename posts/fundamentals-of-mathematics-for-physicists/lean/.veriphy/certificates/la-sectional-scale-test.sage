# name: la-sectional-scale-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MultilinearMaps.sectional_scale_invariant file "la-sectional-scale-test.sage"
# description: (t² r)/(t² q) = r/q for t ≠ 0, at r = q = 0 and explicit rationals, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '1', 'one_ne_zero'], True),
    (['1', '2', '3', '(by norm_num)'], False),
    (['-1 / 2', '5', '-2', '(by norm_num)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
