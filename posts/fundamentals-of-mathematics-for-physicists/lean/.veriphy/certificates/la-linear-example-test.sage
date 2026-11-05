# name: la-linear-example-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearMaps.linear_example file "la-linear-example-test.sage"
# description: (x, y) ↦ (2x − y, 0) is the map of the matrix (2, −1; 0, 0), at the origin and two vectors, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'ext i\nfin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True),
    (['![1, 2]'], False),
    (['![-1 / 2, 3]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
