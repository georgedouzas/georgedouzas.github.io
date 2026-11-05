# name: la-upper-lower-invariant-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Components.upper_lower_invariant file "la-upper-lower-invariant-test.sage"
# description: a'_i u'^i = a_i u^i with u' = A⁻¹ u, a' = a A, at n = 0 and explicit A, u, a (Sage supplies A⁻¹).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '(by simp)', '![]', '![]'], True, 'simp'),
    (['2', '!![1, 0; 1, 1]', '(isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_two_of]))', '![1, 2]', '![3, -1]'], False, 'rw [Matrix.inv_eq_left_inv (B := !![1, 0; -1, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\nnorm_num [Matrix.vecMul, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]'),
    (['3', '!![1, 0, 0; 1, 1, 0; 0, 0, 1]', '(isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_three]))', '![1, 2, 3]', '![1, 0, -1]'], False, 'rw [Matrix.inv_eq_left_inv (B := !![1, 0, 0; -1, 1, 0; 0, 0, 1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\nnorm_num [Matrix.vecMul, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
