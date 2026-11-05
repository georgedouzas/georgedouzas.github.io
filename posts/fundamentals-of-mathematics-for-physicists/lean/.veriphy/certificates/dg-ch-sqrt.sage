# name: dg-ch-sqrt
# technique: square roots of rational squares by named mathlib lemmas, checked in Sage first (transcendental_rewrite recipe)
# goal: Real.sqrt q = p with q = p² rational, p ≥ 0: the lengths read off a Gram matrix (√(1/4) = 1/2) and the
#   height √(1 - (3/5)²) = 4/5 of a point of the hemisphere
# description: Sage confirms the value; Real.sqrt_eq_iff_mul_self_eq reduces it to p * p = q, checked by norm_num.
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
print('Sage: lhs - rhs =', e, '=', e.simplify_full())
assert bool(e.simplify_full() == 0), f'Sage does not confirm the value: lhs - rhs = {e.simplify_full()}'
proof = tactic('rw [Real.sqrt_eq_iff_mul_self_eq (by norm_num) (by norm_num)]\nnorm_num')
