# name: ra-critical-g
# technique: ideal membership (Gröbner cofactors); with no hypothesis, lhs - rhs is the zero polynomial
# goal: x² + 4y² - 2x + 1 = (x-1)² + 4y²
# description: Sage checks lhs - rhs in the ideal of the hypotheses (here: is zero); linear_combination or ring1 checks it.
eqs = [h for h in hyps if h.rel == 'eq' and h.poly is not None]
assert goal is not None and goal.rel == 'eq' and goal.poly is not None, 'the goal must be a polynomial equality'
if eqs:
    assert goal.poly in R.ideal([h.poly for h in eqs]), 'the goal does not follow from the equalities'
    cofactors = goal.poly.lift(R.ideal([h.poly for h in eqs]))
    proof = linear_combination([(c, hyp(problem, h.name)) for c, h in zip(cofactors, eqs)], problem)
else:
    assert goal.poly == 0, f'the identity is false: lhs - rhs = {goal.poly}'
    proof = tactic('ring1')
