# name: de-poly
# technique: ideal membership (Gröbner cofactors), with the empty ideal for an identity (ideal_membership recipe)
# goal: lhs = rhs, a polynomial identity, implied by polynomial equality hypotheses or by algebra alone
# description: Sage writes lhs - rhs as a combination of the equality hypotheses (cofactors by lift) and
#   linear_combination checks it; with no hypothesis Sage checks lhs - rhs = 0 and ring1 checks it again.
eqs = [h for h in hyps if h.rel == 'eq' and h.poly is not None]
assert goal is not None and goal.rel == 'eq', 'the goal must be an equality'
if goal.poly is None:
    assert bool(goal.expr.simplify_full() == 0), f'the identity is false: {goal.expr.simplify_full()}'
    proof = tactic('first | ring1 | (field_simp; done) | (field_simp; ring1) | (norm_num; done) | (simp; done)')
elif not eqs or goal.poly == 0:
    assert goal.poly == 0, f'the identity is false: lhs - rhs = {goal.poly}'
    proof = tactic('ring1')
else:
    assert goal.poly in R.ideal([h.poly for h in eqs]), 'the goal does not follow from the equalities (not in the ideal)'
    cofactors = goal.poly.lift(R.ideal([h.poly for h in eqs]))
    proof = linear_combination([(c, hyp(problem, h.name)) for c, h in zip(cofactors, eqs)], problem)
