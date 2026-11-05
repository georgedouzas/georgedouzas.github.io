# name: ra-inverse-polar-det
# technique: rewrite with named mathlib lemmas, checked in Sage first
# goal: cos φ (cos φ / r) - sin φ (-sin φ / r) = 1/r, the expanded determinant of the inverse polar Jacobian (r ≠ 0)
# description: Sage confirms the identity modulo cos² + sin² = 1; Lean clears r and closes with cos² + sin² = 1.
REWRITES = []  # no table rewrite: the identity is cos² + sin² = 1 after clearing the denominator r
PROOFS = {}
CLOSE = 'field_simp\nlinear_combination Real.cos_sq_add_sin_sq φ'

e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full().canonicalize_radical() == 0) or bool(e.simplify_exp().simplify_trig() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
proof = tactic(CLOSE)
