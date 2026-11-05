# name: cm-poly
# technique: ideal membership (Gröbner cofactors), with the empty ideal for an identity (ideal_membership recipe), and
#   a field_simp closing for identities whose quotients by variables are atoms of the polynomial form
# goal: lhs = rhs, a polynomial or rational identity in the jet variables, implied by polynomial equality hypotheses
#   or by algebra alone
# description: Sage writes lhs - rhs as a combination of the equality hypotheses (cofactors by lift) and
#   linear_combination checks it; with no hypothesis needed Sage checks lhs - rhs = 0 (as a polynomial, or
#   symbolically when quotients occur) and ring1 or field_simp; ring1 checks it again.
eqs = [h for h in hyps if h.rel == 'eq' and h.poly is not None]
assert goal is not None and goal.rel == 'eq', 'the goal must be an equality'
def _sym_zero():
    e = goal.expr
    if e is None:
        return False
    try:
        s = e.simplify_full()
        return bool(s == 0) or bool(s.canonicalize_radical() == 0)
    except Exception:
        return False
if goal.poly is not None and goal.poly == 0:
    proof = tactic('ring1')
elif goal.poly is not None and eqs and goal.poly in R.ideal([h.poly for h in eqs]):
    cofactors = goal.poly.lift(R.ideal([h.poly for h in eqs]))
    proof = linear_combination([(c, hyp(problem, h.name)) for c, h in zip(cofactors, eqs)], problem)
elif _sym_zero():
    proof = tactic('first | ring1 | (field_simp; done) | (field_simp; ring1) | (norm_num; done) | (simp; done) | (field_simp; ring_nf; done)')
else:
    shown = goal.poly if goal.poly is not None else goal.expr
    raise AssertionError(f'the identity does not follow from the equalities: lhs - rhs = {shown}')
