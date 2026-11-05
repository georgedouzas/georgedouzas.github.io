# name: ca-spec-ring
# technique: ideal membership (Gröbner cofactors), the ideal_membership recipe extended to the zero ideal
# goal: lhs = rhs, a polynomial identity in its atoms (possibly implied by polynomial equality hypotheses)
# description: Sage checks lhs - rhs lies in the ideal of the equality hypotheses (the zero ideal when there are
#   none, i.e. the identity holds in the polynomial ring) and finds the cofactors; linear_combination checks them.
eqs = [h for h in hyps if h.rel == 'eq' and h.poly is not None]
assert goal is not None and goal.rel == 'eq', 'the goal must be a polynomial equality'
if goal.poly == 0:
    terms = None
else:
    assert eqs, 'the identity is not a polynomial identity and there are no equality hypotheses'
    assert goal.poly in R.ideal([h.poly for h in eqs]), 'the goal does not follow from the equalities (not in the ideal)'
    cofactors = goal.poly.lift(R.ideal([h.poly for h in eqs]))
    terms = [(c, hyp(problem, h.name)) for c, h in zip(cofactors, eqs) if c != 0]
proof = tactic('ring1') if terms is None else linear_combination(terms, problem)
