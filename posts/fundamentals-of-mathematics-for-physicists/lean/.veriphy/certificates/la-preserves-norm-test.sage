# name: la-preserves-norm-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MinkowskiResults.preserves_norm file "la-preserves-norm-test.sage"
# description: Λᵀ η Λ = η preserves η(v, v), at n = 0 and at rational boosts (cosh u = 5/4, sinh u = 3/4) in 1 + 1 and 1 + 2 dimensions, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0', '(by simp)', '![]'], True, 'simp'),
    (['2', '!![5 / 4, 3 / 4; 3 / 4, 5 / 4]', '!![1, 0; 0, -1]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply])', '![1, 2]'], False),
    (['3', '!![5 / 4, 3 / 4, 0; 3 / 4, 5 / 4, 0; 0, 0, 1]', '!![1, 0, 0; 0, -1, 0; 0, 0, -1]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply])', '![2, 1, -1]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
