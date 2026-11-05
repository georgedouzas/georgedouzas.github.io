# name: ra-steepest-direction-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.steepest_direction file "ra-steepest-direction-test.sage"
# description: ∇f · u ≤ |∇f| for unit u, at (4, 8) with u = (3, 4)/5 (44/5 ≤ 4√5) and other cases
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['4', '8', '3 / 5', '4 / 5', '(by norm_num)'], False, 'have h : (44 / 5 : ℝ) ≤ Real.sqrt 80 := by rw [Real.le_sqrt (by norm_num) (by norm_num)]; norm_num\nnorm_num\ncalc (44 / 5 : ℝ) ≤ Real.sqrt 80 := h\n  _ = _ := by norm_num'), (['0', '0', '1', '0', '(by norm_num)'], True, 'norm_num'), (['3', '4', '3 / 5', '4 / 5', '(by norm_num)'], False, 'norm_num\ntry rw [show (25 : ℝ) = 5 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
