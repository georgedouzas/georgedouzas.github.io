# name: la-cross-contraction-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Contraction.elementary_cross_contraction file "la-cross-contraction-test.sage"
# description: u^i v^j α_j β_i = β(u) α(v) at n = 0 and explicit vectors and covectors, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [dotProduct, Fin.sum_univ_succ]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '![]', '![]', '![]', '![]'], True, 'simp'),
    (['2', '![1, 2]', '![3, -1]', '![0, 1]', '![2, 2]'], False),
    (['3', '![1, 0, -1]', '![1, 2, 3]', '![2, 1, 0]', '![0, 1, 1]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
