# name: dg-cc2-deriv
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value
# goal: deriv (fun t => f t) a = v for the explicit functions of the Differential geometry chapter (part cc2):
#   -sin t cos t, cos t / sin t, cos t / (2 sin t), exp (-t), -exp (-t), R + r cos (t / r), -sin (t / r)
# description: Sage differentiates and checks the stated value; mathlib's derivative lemmas, composed, prove HasDerivAt
#   at the (symbolic) point; the nonzero conditions (sin t, r) are hypotheses of the theorems.
PROOFS = {}
assert problem.derivatives, 'the goal has no derivative Veriphy translates; inspect it with #veriphy_translate'
d = problem.derivatives[0]
value = diff(d.body, d.var).subs({d.var: d.point})
assert bool(goal.expr.simplify_full() == 0), f'Sage computes the derivative as {value}, not the stated value'
proof = steps(
    has_deriv_at_proof(d.body, d.var, d.point, value, problem, proofs=PROOFS, function=d.function, at=d.at),
    tactic('exact hd.deriv.trans (by first | rfl | ring1 | (field_simp; ring1) | norm_num)'),
)
