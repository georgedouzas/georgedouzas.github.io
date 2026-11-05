# name: set-trig-periodic
# technique: periodicity of cos by named mathlib lemmas, checked in Sage first (transcendental_rewrite recipe)
# goal: cos(2(φ + 2π)) = cos(2φ) and cos((φ + 2π)/2) = -cos(φ/2): which functions of an angle descend to the circle
# description: Sage confirms the identity; the argument is rewritten as 2φ + 2·2π (Real.cos_add_nat_mul_two_pi) or as
#   φ/2 + π (Real.cos_add_pi).
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
print('Sage: lhs - rhs =', e, '=', e.simplify_full())
assert bool(e.simplify_full() == 0) or bool(e.trig_reduce().simplify_full() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
phi = [t for t in atoms.values() if 'Real.pi' not in t and 'cos' not in t][0] if atoms else 'φ'
if '-Real.cos' in goal.text or '= -' in goal.text:
    proof = tactic(f'rw [show ({phi} + 2 * Real.pi) / 2 = {phi} / 2 + Real.pi by ring, Real.cos_add_pi]')
else:
    proof = tactic(f'rw [show 2 * ({phi} + 2 * Real.pi) = 2 * {phi} + ((2 : ℕ) : ℝ) * (2 * Real.pi) by push_cast; ring, '
                   'Real.cos_add_nat_mul_two_pi]')
