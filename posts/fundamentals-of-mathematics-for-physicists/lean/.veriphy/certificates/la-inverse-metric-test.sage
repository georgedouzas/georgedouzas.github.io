# name: la-inverse-metric-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MixedTensors.inverse_metric file "la-inverse-metric-test.sage"
# description: g^{ij} g_{jk} = δ^i_k at n = 0, the Minkowski metric of 1 + 1 dimensions and a positive 3 × 3 metric (Sage supplies g⁻¹).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '(by simp)'], True, 'ext i j\nexact i.elim0'),
    (['2', '!![1, 0; 0, -1]', '(by norm_num [Matrix.det_fin_two_of])'], False, 'rw [Matrix.inv_eq_left_inv (B := !![1, 0; 0, -1]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\next i j\nfin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]'),
    (['3', '!![2, 1, 0; 1, 2, 1; 0, 1, 2]', '(by norm_num [Matrix.det_fin_three])'], False, 'rw [Matrix.inv_eq_left_inv (B := !![(3 / 4), (-1 / 2), (1 / 4); (-1 / 2), 1, (-1 / 2); (1 / 4), (-1 / 2), (3 / 4)]) (by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])]\next i j\nfin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
