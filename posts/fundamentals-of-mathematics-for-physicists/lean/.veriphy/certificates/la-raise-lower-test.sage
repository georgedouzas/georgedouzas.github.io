# name: la-raise-lower-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.NondegenerateForms.raise_lower file "la-raise-lower-test.sage"
# description: raising undoes lowering, B⁻¹(B v) = v, at n = 0, at the Gram matrix of f and at η (Sage supplies B⁻¹).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '(by simp)', '![]'], True, 'simp'),
    (['2', '!![5, 4; 4, 5]', '(by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])', '![2, 1]'], False, 'rw [Matrix.inv_eq_left_inv (B := !![(5 / 9), (-4 / 9); (-4 / 9), (5 / 9)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\nnorm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]'),
    (['4', '!![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1]', '(by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])', '![2, 1, 0, 3]'], False, 'rw [Matrix.inv_eq_left_inv (B := !![1, 0, 0, 0; 0, -1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\nnorm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
