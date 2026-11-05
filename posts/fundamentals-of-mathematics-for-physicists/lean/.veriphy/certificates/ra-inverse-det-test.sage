# name: ra-inverse-det-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.inverse_det file "ra-inverse-det-test.sage"
# description: det B = 1/det A when B A = 1, at explicit pairs
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['1', '1', '(by simp)'], True, 'simp'), (['!![(2 : ℝ), 1; 1, 1]', '!![(1 : ℝ), -1; -1, 2]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])'], False, 'rw [Matrix.det_fin_two_of, Matrix.det_fin_two_of]; norm_num'), (['!![(2 : ℝ), 0; 0, 4]', '!![(1 / 2 : ℝ), 0; 0, 1 / 4]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])'], False, 'rw [Matrix.det_fin_two_of, Matrix.det_fin_two_of]; norm_num')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
