# name: dg-vf2-trig-values
# technique: values of sin and cos at the special angles π/6, π/3, π/2, π and 2π, checked exactly in Sage and closed in
#   Lean by mathlib's value lemmas (transcendental_rewrite recipe, adapted: the rewrite list is the fixed table of
#   value lemmas, as in fa-matrix-values.sage)
# goal: an identity between numbers, √2, √3 and sin, cos at multiples of π/6 and π/4
# description: Sage checks that lhs - rhs is exactly 0; Lean rewrites the values (Real.sin_pi_div_three, …) and
#   closes with norm_num or ring1.
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
assert bool(e.simplify_full() == 0) or bool(e.canonicalize_radical() == 0), f'the identity is false: {e.simplify_full()}'
VALUES = ['Real.sin_pi_div_six', 'Real.cos_pi_div_six', 'Real.sin_pi_div_three', 'Real.cos_pi_div_three',
          'Real.sin_pi_div_four', 'Real.cos_pi_div_four', 'Real.sin_pi_div_two', 'Real.cos_pi_div_two', 'Real.sin_pi',
          'Real.cos_pi', 'Real.sin_two_pi', 'Real.cos_two_pi', 'Real.sin_zero', 'Real.cos_zero']
proof = tactic(f'simp only [{", ".join(VALUES)}]\nall_goals first | ring1 | (norm_num; done) | (norm_num; ring1)')
