# name: la-det-diagonal-prod-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MinkowskiResults.det_diagonal_prod file "la-det-diagonal-prod-test.sage"
# description: det diag(d) = Π dᵢ at n = 0 and explicit diagonals of size 2, 3, 4 (Laplace expansion by evaluation).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.prod_univ_succ, Matrix.diagonal_apply, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc]\n<;> norm_num'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '![]'], True, 'simp'),
    (['2', '![2, 3]'], False),
    (['3', '![1, -1, 1 / 2]'], False),
    (['4', '![1, -1, -1, -1]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
