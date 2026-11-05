# name: set-sqrt-three-quarters
# technique: square roots by named mathlib lemmas, checked in Sage first (transcendental_rewrite recipe)
# goal: Real.sqrt (1 - (1 / 2) ^ 2) = Real.sqrt 3 / 2, the X-coordinate of the points of the circle at height ½
# description: Sage confirms the value; rewriting 1 - ¼ = 3/2², Real.sqrt_div' and Real.sqrt_sq close it.
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
print('Sage: lhs - rhs =', e, '=', e.simplify_full())
assert bool(e.simplify_full() == 0), f'Sage does not confirm the value: lhs - rhs = {e.simplify_full()}'
proof = tactic("rw [show (1 : ℝ) - (1 / 2) ^ 2 = 3 / 2 ^ 2 by norm_num, Real.sqrt_div' _ (by norm_num), "
               "Real.sqrt_sq (by norm_num)]")
