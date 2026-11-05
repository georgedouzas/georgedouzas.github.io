# name: dg-ch-poly
# technique: ideal membership (Gröbner cofactors); with no hypothesis, lhs - rhs is the zero polynomial
# goal: lhs = rhs, a polynomial identity in the atoms (trigonometric values generalised to variables, with their
#   Pythagorean relations as hypotheses), after the denominators have been cleared by field_simp
# description: Sage checks lhs - rhs in the ideal of the hypotheses (or that it is the zero polynomial);
#   linear_combination with the cofactors, or ring1, checks it.
eqs = [h for h in hyps if h.rel == 'eq' and h.poly is not None]
assert goal is not None and goal.rel == 'eq' and goal.poly is not None, 'the goal must be a polynomial equality'
if eqs and goal.poly != 0:
    assert goal.poly in R.ideal([h.poly for h in eqs]), f'the goal does not follow from the equalities: {goal.poly}'
    cofactors = goal.poly.lift(R.ideal([h.poly for h in eqs]))
    proof = linear_combination([(c, hyp(problem, h.name)) for c, h in zip(cofactors, eqs)], problem)
else:
    assert goal.poly == 0, f'the identity is false: lhs - rhs = {goal.poly}'
    proof = tactic('ring1')
