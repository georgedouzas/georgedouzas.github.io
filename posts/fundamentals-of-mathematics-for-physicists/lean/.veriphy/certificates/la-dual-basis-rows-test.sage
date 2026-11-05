# name: la-dual-basis-rows-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualResults.dual_basis_rows file "la-dual-basis-rows-test.sage"
# description: the rows of B⁻¹ are the dual basis (B⁻¹ B = 1) at n = 0 and at the bases f and b (Sage supplies B⁻¹).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '(by simp)'], True, 'ext i j\nexact i.elim0'),
    (['2', '!![2, 1; 1, 2]', '(by norm_num [Matrix.det_fin_two_of])'], False, 'rw [Matrix.inv_eq_left_inv (B := !![(2 / 3), (-1 / 3); (-1 / 3), (2 / 3)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\next i j\nfin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]'),
    (['3', '!![1, 0, 1; 1, 1, 0; 0, 1, 1]', '(by norm_num [Matrix.det_fin_three])'], False, 'rw [Matrix.inv_eq_left_inv (B := !![(1 / 2), (1 / 2), (-1 / 2); (-1 / 2), (1 / 2), (1 / 2); (1 / 2), (-1 / 2), (1 / 2)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\next i j\nfin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
