# name: la-beta-level-set-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearFunctionals.beta_level_set file "la-beta-level-set-test.sage"
# description: β(p + w) = c for β(p) = c and w in the kernel, at explicit p and kernel vectors w, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rw [beta_apply]\nnorm_num'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0', '(by rw [beta_apply]; norm_num)', '(by rw [LinearMap.mem_ker, beta_apply]; norm_num)'], True),
    (['![1, 0, 0]', '![1, 3, 0]', '3', '(by rw [beta_apply]; norm_num)', '(by rw [LinearMap.mem_ker, beta_apply]; norm_num)'], False),
    (['![0, 0, 1]', '![0, 2, 1]', '2', '(by rw [beta_apply]; norm_num)', '(by rw [LinearMap.mem_ker, beta_apply]; norm_num)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
