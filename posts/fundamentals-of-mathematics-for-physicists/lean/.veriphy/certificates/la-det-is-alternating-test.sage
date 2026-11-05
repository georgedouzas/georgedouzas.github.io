# name: la-det-is-alternating-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MultilinearMaps.det_is_alternating file "la-det-is-alternating-test.sage"
# description: the alternating map of the rows evaluates to det A, at n = 0 and explicit 2 × 2 and 3 × 3 matrices.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rfl'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0'], True),
    (['2', '!![1, 2; 3, 4]'], False),
    (['3', '!![2, 1, 0; 1, 2, 1; 0, 1, 2]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
