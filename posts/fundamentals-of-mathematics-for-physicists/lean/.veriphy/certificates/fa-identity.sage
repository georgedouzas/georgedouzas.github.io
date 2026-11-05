# name: fa-identity
# technique: ideal membership (ideal_membership recipe) with no hypotheses: lhs - rhs lies in the zero ideal
# goal: lhs = rhs, an identity of rational functions in π and the variables (normalisations, Parseval sums)
# description: Sage simplifies lhs - rhs to 0 (the goal is in the zero ideal); Lean checks it by ring normalisation,
#   after clearing the nonzero denominators (π, factorials) with field_simp.
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
assert bool(e.simplify_full() == 0), f'the identity is false: lhs - rhs = {e.simplify_full()}'
proof = tactic('first | ring1 | (field_simp; ring1) | (have hπ : Real.pi ≠ 0 := Real.pi_ne_zero; field_simp; ring1)')
