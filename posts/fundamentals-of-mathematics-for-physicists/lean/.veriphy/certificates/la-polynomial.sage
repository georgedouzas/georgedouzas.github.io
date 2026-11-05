# name: la-polynomial
# technique: ideal membership in the zero ideal (ideal_membership recipe without hypotheses), read from the goal
# goal: lhs = rhs in a polynomial ring ℝ[X] or ℚ[X] (Polynomial.X as the generator, constants C c, eval, derivative),
#   with no hypotheses
# description: Sage expands lhs - rhs from the goal and checks it is the zero polynomial (membership in the zero ideal);
#   Lean checks it by ring normalisation (linear_combination with no hypothesis), after pushing C through sums or
#   evaluating eval and derivative.
assert goal is not None and goal.rel == 'eq', 'the goal must be a polynomial equality'
assert goal.expr is not None, 'the goal has no symbolic form'
assert bool(goal.expr.expand() == 0), f'the identity is false: lhs - rhs = {goal.expr.expand()}'
proof = tactic(
    'first\n'
    '| linear_combination\n'
    '| (simp only [map_add, map_sub, map_mul, map_neg]; ring)\n'
    '| (norm_num; done)\n'
    '| (simp [Polynomial.derivative_X_pow]; norm_num [map_ofNat]; done)'
)
