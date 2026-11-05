# name: ca-res-ring
# technique: calc derivation (one step), the identity confirmed in Sage first
# goal: lhs = rhs, a polynomial or rational identity over ℂ or ℝ (nonzero denominators stated as hypotheses)
# description: Sage confirms that lhs - rhs simplifies to 0; the calc step closes by ring1, or by field_simp then ring1.
assert goal is not None and goal.rel == 'eq', 'the goal must be an equation lhs = rhs'
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full() == 0) or bool(e.expand().simplify_rational() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
chain = [('=', term(rhs.text), tactic('first | ring1 | (field_simp; done) | (field_simp; ring1)'))]
proof = calc_chain(term(lhs.text), chain, problem)
