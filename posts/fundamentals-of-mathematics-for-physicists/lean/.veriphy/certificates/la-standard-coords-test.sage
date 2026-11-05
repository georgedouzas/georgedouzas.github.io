# name: la-standard-coords-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Bases.standard_coords file "la-standard-coords-test.sage"
# description: the coordinates of x in the standard basis of ℝⁿ are its entries, at n = 1, 3, 4, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['1', '![5]', '0'], True),
    (['3', '![5, 7, 9]', '1'], False),
    (['4', '![1, -2, 1 / 2, 3]', '3'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
