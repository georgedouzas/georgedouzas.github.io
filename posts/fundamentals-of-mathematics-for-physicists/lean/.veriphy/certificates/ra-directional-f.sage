# name: ra-directional-f
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value
# goal: d/dt f((1, 0) + t(3, 4)/5) at 0 = 8/5
# description: Sage differentiates; mathlib's derivative lemmas, composed, prove HasDerivAt; deriv follows.
PROOFS = {}  # EDIT: tactics proving the nonzero conditions no hypothesis states, by their Lean text
assert problem.derivatives, 'the goal has no derivative Veriphy translates; inspect it with #veriphy_translate'
d = problem.derivatives[0]
value = diff(d.body, d.var).subs({d.var: d.point})
assert bool(goal.expr.simplify_full() == 0), f'Sage computes the derivative as {value}, not the stated value'
proof = steps(
    has_deriv_at_proof(d.body, d.var, d.point, value, problem, proofs=PROOFS, function=d.function, at=d.at),
    tactic('exact hd.deriv.trans (by first | rfl | ring1 | (field_simp; ring1) | norm_num)'),
)
