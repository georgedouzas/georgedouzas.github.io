# name: pr-identity
# technique: ideal membership in the zero ideal (ideal_membership recipe without hypotheses), read from the goal
# goal: lhs = rhs, a polynomial or rational-function identity in real or rational atoms with no hypotheses needed
#   beyond the non-vanishing of denominators (the algebraic identities of the probability chapter: cancellations,
#   moment and cumulant formulas, variances of weighted means)
# description: Sage simplifies lhs - rhs to zero (membership in the zero ideal after clearing denominators); Lean
#   checks it by ring normalisation (linear_combination with no hypothesis), after field_simp when denominators occur.
assert goal is not None and goal.rel == 'eq', 'the goal must be an equality'
assert goal.expr is not None, 'the goal has no symbolic form'
assert bool(goal.expr.simplify_full() == 0), f'the identity is false: lhs - rhs = {goal.expr.simplify_full()}'
proof = tactic(
    'first\n'
    '| linear_combination\n'
    '| (field_simp; ring1)\n'
    '| (norm_num; done)\n'
    '| (field_simp; norm_num; done)'
)
