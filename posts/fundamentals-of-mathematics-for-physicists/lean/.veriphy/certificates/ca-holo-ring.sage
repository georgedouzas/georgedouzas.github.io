# name: ca-holo-ring
# technique: ideal membership (Gröbner cofactors), the ideal_membership recipe extended to pure ring identities and
#   to goals over ℂ that Veriphy 0.3.0 does not translate
# goal: lhs = rhs over ℂ or ℝ, a polynomial identity in its atoms (Complex.I, conj z, ↑x, exp terms generalised to
#   variables, ...), possibly implied by polynomial equality hypotheses such as Complex.I ^ 2 = -1 or s ^ 2 = 2. When
#   the goal is not translated (a ℂ-valued goal with numerals), the tactic `ca_goal_hyp` first records it as the
#   hypothesis `hgoal : lhs - rhs = G`, which Veriphy does translate; the script reads the goal from it.
# description: Sage writes lhs - rhs as a combination of the equality hypotheses (cofactors by lift; none when the
#   identity holds in the polynomial ring itself); linear_combination (or ring1, for an identity of the polynomial
#   ring) checks it in Lean.
hg = [h for h in hyps if h.name == 'hgoal' and h.poly is not None]
eqs = [h for h in hyps if h.rel == 'eq' and h.poly is not None and h.name != 'hgoal']
if goal is not None and goal.rel == 'eq' and goal.poly is not None:
    target = goal.poly
else:
    assert hg, 'the goal is not a translated polynomial equality; record it with ca_goal_hyp first'
    gvar = [R(k) for k, v in atoms.items() if v == 'G']
    assert len(gvar) == 1, f'no atom G in {atoms}'
    target = hg[0].poly + gvar[0]
if eqs and target != 0:
    assert target in R.ideal([h.poly for h in eqs]), 'the goal does not follow from the equalities (not in the ideal)'
    cofactors = target.lift(R.ideal([h.poly for h in eqs]))
    terms = [(c, hyp(problem, h.name)) for c, h in zip(cofactors, eqs) if c != 0]
else:
    assert target == 0, f'the identity is false in the polynomial ring: lhs - rhs = {target}'
    terms = []
proof = linear_combination(terms, problem) if terms else tactic('ring1')
