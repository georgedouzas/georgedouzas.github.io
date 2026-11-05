# name: cm-escape-speed-iff-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Energy.escape_speed_iff file "cm-escape-speed-iff-test.sage"
# description: E ≥ 0 ↔ v ≥ √(2GM/R) at (G, M, m, R, v) = (0, 1, 1, 1, 0) (edge), (1, 1, 1, 2, 3), (2, 1, 1, 1, 1)
SEED = 0  # the instances are listed by hand
CASES = [(['0', '1', '1', '1', '0', '(by norm_num)', '(by norm_num)', '(by norm_num)', '(by norm_num)'], True, 'norm_num'),
         (['1', '1', '1', '2', '3', '(by norm_num)', '(by norm_num)', '(by norm_num)', '(by norm_num)'], False,
          'rw [show (2 : ℝ) * 1 * 1 / 2 = 1 by norm_num, Real.sqrt_one]\nnorm_num'),
         (['2', '1', '1', '1', '1', '(by norm_num)', '(by norm_num)', '(by norm_num)', '(by norm_num)'], False,
          'rw [show (2 : ℝ) * 2 * 1 / 1 = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]\nnorm_num')]
instances = [kernel_instance(*values, proof=p, edge=edge) for values, edge, p in CASES]
