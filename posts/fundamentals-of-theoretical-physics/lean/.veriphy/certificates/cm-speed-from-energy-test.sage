# name: cm-speed-from-energy-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Energy.speed_from_energy file "cm-speed-from-energy-test.sage"
# description: v = √(2(E - V)/m) at (m, E, V, v) = (1, 1/2, 0, 1) (edge), (1, 3, 1, 2), (2, 8, -1, 3)
SEED = 0  # the instances are listed by hand
CASES = [(['1', '1/2', '0', '1', '(by norm_num)', '(by norm_num)', '(by norm_num)'], True, 'norm_num'),
         (['1', '3', '1', '2', '(by norm_num)', '(by norm_num)', '(by norm_num)'], False,
          'rw [show (2 : ℝ) * (3 - 1) / 1 = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]'),
         (['2', '8', '-1', '3', '(by norm_num)', '(by norm_num)', '(by norm_num)'], False,
          'rw [show (2 : ℝ) * (8 - -1) / 2 = 3 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]')]
instances = [kernel_instance(*values, proof=p, edge=edge) for values, edge, p in CASES]
