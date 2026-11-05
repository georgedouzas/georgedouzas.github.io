# name: cm-projectile-range-max-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Momentum.projectile_range_max file "cm-projectile-range-max-test.sage"
# description: the range v₀² sin 2α / g is at most v₀²/g, attained at α = π/4, at α = 0 (edge), π/4, π/2
SEED = 0  # the instances are listed by hand
PROOF = ('constructor <;> norm_num [show (2 : ℝ) * (Real.pi / 4) = Real.pi / 2 by ring, '
         'show (2 : ℝ) * (Real.pi / 2) = Real.pi by ring, Real.sin_pi_div_two, Real.sin_pi]')
CASES = [(['1', '0', '1', '(by norm_num)'], True), (['2', 'Real.pi / 4', '1', '(by norm_num)'], False),
         (['1', 'Real.pi / 2', '3', '(by norm_num)'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]
