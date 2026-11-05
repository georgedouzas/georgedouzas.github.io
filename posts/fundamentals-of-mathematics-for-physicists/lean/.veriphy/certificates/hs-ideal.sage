# name: hs-ideal
# technique: polynomial identity by ring normalisation (ideal_membership recipe, no hypotheses: the goal itself is
#   checked as a polynomial identity)
# goal: lhs = rhs, a polynomial identity in the free variables of the goal
# description: Sage expands lhs - rhs to 0; Lean closes by ring1.
assert goal is not None and goal.rel == 'eq', 'the goal must be a polynomial equality'
assert goal.poly == 0, f'the identity is false: lhs - rhs = {goal.poly}'
proof = tactic('ring1')
