# name: la-det-mixed-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Components.det_mixed file "la-det-mixed-test.sage"
# description: det (A⁻¹ T A) = det T at n = 0 and explicit A, T (Sage supplies A⁻¹).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0', '(by simp)'], True, 'simp'),
    (['2', '!![1, 0; 1, 1]', '!![1, 3; 0, 2]', '(isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_two_of]))'], False, 'rw [Matrix.inv_eq_left_inv (B := !![1, 0; -1, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\nsimp [Matrix.det_fin_two, Matrix.det_fin_three, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]\n<;> norm_num'),
    (['3', '!![1, 0, 0; 1, 1, 0; 0, 0, 1]', '!![1, 2, 0; 0, 3, 1; 0, 0, 5]', '(isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_three]))'], False, 'rw [Matrix.inv_eq_left_inv (B := !![1, 0, 0; -1, 1, 0; 0, 0, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\nsimp [Matrix.det_fin_two, Matrix.det_fin_three, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]\n<;> norm_num'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
