# name: la-coordinate-functional-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearFunctionals.coordinate_functional file "la-coordinate-functional-test.sage"
# description: the i-th coordinate functional returns v^i, at n = 1, 3, 4, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rfl'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['1', '0', '![5]'], True),
    (['3', '1', '![5, 7, 9]'], False),
    (['4', '3', '![1, -2, 1 / 2, 3]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
