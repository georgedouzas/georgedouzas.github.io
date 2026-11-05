# name: dg-cc1-values
# technique: trigonometric values at special angles (transcendental_rewrite recipe, adapted): Sage evaluates the
#   identity exactly, Lean rewrites with mathlib's value lemmas (cos 0, sin π, cos (π/2), cos (π/3), …) and norm_num
# goal: an identity between expressions whose transcendental parts are sin and cos at 0, π/6, π/3, π/2, π, 2π,
#   possibly scaled by symbolic constants (e.g. 2π cos (π/2) = 0, cos (2π cos (π/3)) = -1)
# description: Sage checks lhs - rhs simplifies to 0 exactly; Lean closes with simp only [the value lemmas] and norm_num.
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
entries = [SR(x) for x in e.list()] if hasattr(e, 'nrows') else [SR(e)]
bad = [x for x in entries if not bool(x.simplify_full().simplify_trig() == 0)]
assert not bad, f'the identity is false: nonzero entries {bad}'
VALUES = ('Real.cos_zero, Real.sin_zero, Real.cos_pi, Real.sin_pi, Real.cos_two_pi, Real.sin_two_pi, '
          'Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi_div_three, Real.sin_pi_div_three, '
          'Real.cos_pi_div_six, Real.sin_pi_div_six')
proof = tactic(f'first\n  | (simp only [{VALUES}]; done)\n  | (simp only [{VALUES}]; norm_num; done)\n'
               f'  | (norm_num [{VALUES}]; done)\n'
               f'  | (rw [show (2 : ℝ) * Real.pi * Real.cos (Real.pi / 3) = Real.pi by rw [Real.cos_pi_div_three]; ring]; '
               f'simp only [{VALUES}]; norm_num; done)\n'
               f'  | (ext i; fin_cases i <;> simp [{VALUES}] <;> norm_num)\n'
               f'  | (ext i j; fin_cases i <;> fin_cases j <;> simp [{VALUES}, Matrix.one_apply] <;> norm_num)')
