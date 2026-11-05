# name: la-alpha-level-sets-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearFunctionals.alpha_level_sets file "la-alpha-level-sets-test.sage"
# description: α(v + w) = α(v) for w in the kernel, at explicit v and kernel vectors w, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [alpha]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '(by rw [LinearMap.mem_ker]; norm_num [alpha])'], True),
    (['![1, 2]', '![1, -2]', '(by rw [LinearMap.mem_ker]; norm_num [alpha])'], False),
    (['![-3, 1 / 2]', '![-1 / 2, 1]', '(by rw [LinearMap.mem_ker]; norm_num [alpha])'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
