# name: la-t-linear-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MultilinearMaps.T_linear file "la-t-linear-test.sage"
# description: T(2u + v, v, w) = 2 T(u, v, w) + T(v, v, w) at the zero vectors and explicit vectors, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [T]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0'], True),
    (['![1, 2, 3]', '![0, 1, -1]', '![2, 0, 1]'], False),
    (['![1 / 2, -1, 0]', '![3, 0, 1]', '![1, 1, 1]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
