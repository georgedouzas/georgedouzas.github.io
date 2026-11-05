# name: dg-ch-trig-values
# technique: special values of cos and sin by named mathlib lemmas, checked in Sage first (transcendental_rewrite recipe)
# goal: an equation between cos, sin (and sqrt) at 0, ±π/2, π, 3π/2, 2π, 4π, π/3, π/6, 5π/3 and rational values: the
#   angles read off the circle, the poles of the sphere, the coordinate curves at special latitudes
# description: Sage evaluates both sides exactly; simp with the special-value lemmas (the angle rewritten as a sum
#   with π or 2π where needed) and norm_num close the goal.
LEMMAS = ['Real.cos_pi_div_three', 'Real.sin_pi_div_three', 'Real.cos_pi_div_two', 'Real.sin_pi_div_two',
          'Real.cos_pi_div_six', 'Real.sin_pi_div_six', 'Real.cos_zero', 'Real.sin_zero', 'Real.cos_pi',
          'Real.sin_pi', 'Real.cos_two_pi', 'Real.sin_two_pi', 'Real.cos_neg', 'Real.sin_neg',
          'Real.cos_add_pi', 'Real.sin_add_pi', 'Real.cos_two_pi_sub', 'Real.sin_two_pi_sub',
          'Real.cos_add_two_pi', 'Real.sin_add_two_pi', 'Real.cos_nat_mul_pi', 'Real.sin_nat_mul_pi']
REWRITES = ['show (3 : ℝ) * Real.pi / 2 = Real.pi / 2 + Real.pi by ring',
            'show (5 : ℝ) * Real.pi / 3 = 2 * Real.pi - Real.pi / 3 by ring',
            'show (4 : ℝ) * Real.pi = 2 * Real.pi + 2 * Real.pi by ring',
            'show (2 : ℝ) * (2 * Real.pi) = 2 * Real.pi + 2 * Real.pi by ring']
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
print('Sage: lhs - rhs =', e, '=', e.simplify_full())
assert bool(e.simplify_full() == 0), f'Sage does not confirm the value: lhs - rhs = {e.simplify_full()}'
proof = tactic('simp only [' + ', '.join(REWRITES) + '] at *\nsimp [' + ', '.join(LEMMAS) + ']\n'
               'all_goals (first | done | norm_num [' + ', '.join(LEMMAS) + '] | (ring_nf; done) | (ring_nf; norm_num [' + ', '.join(LEMMAS) + ']))')
