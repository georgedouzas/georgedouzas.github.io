# name: la-coords-inverse-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaceResults.coords_inverse file "la-coords-inverse-test.sage"
# description: λ = F⁻¹ v when F λ = v, at n = 0 and at the bases f of ℚ² and b of ℚ³ (Sage supplies F⁻¹, Lean checks F⁻¹ F = 1).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '(by simp)', '![]', '![]', '(by ext i; exact i.elim0)'], True, 'ext i\nexact i.elim0'),
    (['2', '!![2, 1; 1, 2]', '(isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_two_of]))', '![2, 1]', '![5, 4]', '(by ext i; fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ])'], False, 'rw [Matrix.inv_eq_left_inv (B := !![(2 / 3), (-1 / 3); (-1 / 3), (2 / 3)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\next i\nfin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]'),
    (['3', '!![1, 0, 1; 1, 1, 0; 0, 1, 1]', '(isUnit_iff_ne_zero.mpr (by norm_num [Matrix.det_fin_three]))', '![1 / 2, 5 / 2, 3 / 2]', '![2, 3, 4]', '(by ext i; fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ])'], False, 'rw [Matrix.inv_eq_left_inv (B := !![(1 / 2), (1 / 2), (-1 / 2); (-1 / 2), (1 / 2), (1 / 2); (1 / 2), (-1 / 2), (1 / 2)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\next i\nfin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
