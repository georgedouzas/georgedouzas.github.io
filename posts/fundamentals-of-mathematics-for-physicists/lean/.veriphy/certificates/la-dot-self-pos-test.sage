# name: la-dot-self-pos-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Signature.dot_self_pos file "la-dot-self-pos-test.sage"
# description: v · v > 0 for v ≠ 0, at a unit vector of ℝ¹ and explicit vectors of ℝ² and ℝ³, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [dotProduct, Fin.sum_univ_succ]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['1', '![1]', '(by intro h; have := congrFun h 0; norm_num at this)'], True),
    (['2', '![1, -2]', '(by intro h; have := congrFun h 0; norm_num at this)'], False),
    (['3', '![1 / 2, 0, 3]', '(by intro h; have := congrFun h 0; norm_num at this)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
