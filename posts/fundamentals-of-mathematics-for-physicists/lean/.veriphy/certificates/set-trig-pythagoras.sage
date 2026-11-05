# name: set-trig-pythagoras
# technique: rewrite with a named mathlib lemma, checked in Sage first (transcendental_rewrite recipe)
# goal: Real.sin x ^ 2 + 0 ^ 2 + Real.cos x ^ 2 = 1 (the great-circle path γ(t) = (sin(πt/2), 0, cos(πt/2)) stays on the
#   sphere)
# description: Sage confirms the identity; Real.sin_sq_add_cos_sq closes it after dropping 0².
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
print('Sage: lhs - rhs =', e, '=', e.simplify_full())
assert bool(e.simplify_full() == 0) or bool(e.simplify_trig() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
proof = tactic('simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, add_zero, Real.sin_sq_add_cos_sq]')
