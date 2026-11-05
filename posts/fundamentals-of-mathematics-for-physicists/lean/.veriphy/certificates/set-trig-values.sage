# name: set-trig-values
# technique: special values of cos and sin by named mathlib lemmas, checked in Sage first (transcendental_rewrite recipe)
# goal: an equation between cos, sin at 0, π/3, π/2 (and products with them) and rational or √3 values: the point of
#   the torus with angles (π/3, π/2), cos(π/3) = ½, the endpoints of a great-circle path
# description: Sage evaluates both sides exactly; norm_num with the special-value lemmas closes the goal.
LEMMAS = ['Real.cos_pi_div_three', 'Real.sin_pi_div_three', 'Real.cos_pi_div_two', 'Real.sin_pi_div_two',
          'Real.cos_zero', 'Real.sin_zero']
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
print('Sage: lhs - rhs =', e, '=', e.simplify_full())
assert bool(e.simplify_full() == 0), f'Sage does not confirm the value: lhs - rhs = {e.simplify_full()}'
proof = tactic('norm_num [' + ', '.join(LEMMAS) + '] <;> ring_nf')
