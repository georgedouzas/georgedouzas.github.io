# name: ideal_membership
# technique: ideal membership (Gröbner cofactors)
# goal: lhs = rhs, implied by polynomial equality hypotheses
# description: Write lhs - rhs as a combination of the equality hypotheses; linear_combination checks it.
eqs = [h for h in hyps if h.rel == 'eq' and h.poly is not None]
assert goal is not None and goal.rel == 'eq', 'the goal must be a polynomial equality'
assert eqs, 'no polynomial equality hypotheses to combine'
assert goal.poly in R.ideal([h.poly for h in eqs]), 'the goal does not follow from the equalities (not in the ideal)'
cofactors = goal.poly.lift(R.ideal([h.poly for h in eqs]))
proof = linear_combination([(c, hyp(problem, h.name)) for c, h in zip(cofactors, eqs)], problem)
