# name: ca-num-ring
# technique: ideal membership in the zero ideal (ideal_membership recipe): Sage expands lhs - rhs to zero
# goal: lhs = rhs, a polynomial identity (numeral denominators allowed) that needs no hypothesis
# description: Sage checks that lhs - rhs is the zero polynomial in the goal's atoms; ring1 checks it in Lean.
assert goal is not None and goal.rel == 'eq', 'the goal must be an equation'
assert goal.poly is not None, f'the goal has no polynomial form: {goal.text}'
assert goal.poly == 0, f'lhs - rhs does not expand to zero: {goal.poly}'
proof = tactic('ring1')
