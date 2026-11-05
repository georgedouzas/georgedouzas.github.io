# name: la-product-functionals-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MultilinearMaps.product_functionals_linear file "la-product-functionals-test.sage"
# description: α(u + t u') β(v) γ(w) is linear in the first slot, at n = 0 and explicit covectors and vectors, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [dotProduct, Fin.sum_univ_succ]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '![]', '![]', '![]', '![]', '![]', '![]', '![]', '0'], True, 'simp'),
    (['2', '![1, 2]', '![0, 1]', '![3, -1]', '![1, 1]', '![2, 0]', '![1, -1]', '![0, 5]', '2'], False),
    (['3', '![1, 0, -1]', '![2, 1, 0]', '![0, 0, 1]', '![1, 2, 3]', '![0, 1, -1]', '![2, 0, 1]', '![1, 1, 1]', '-1 / 2'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
