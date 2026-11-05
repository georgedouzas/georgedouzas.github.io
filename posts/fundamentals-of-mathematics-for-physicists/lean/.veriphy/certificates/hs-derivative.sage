# name: hs-derivative
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value (derivative recipe)
# goal: deriv (fun x => f x) a = v, f built from powers, exp, sin, cos
# description: Sage differentiates; mathlib's derivative lemmas, composed, prove HasDerivAt at the point; deriv follows.
PROOFS = {}
assert problem.derivatives, 'the goal has no derivative Veriphy translates; inspect it with #veriphy_translate'
d = problem.derivatives[0]
value = diff(d.body, d.var).subs({d.var: d.point})
assert bool(goal.expr.simplify_full() == 0), f'Sage computes the derivative as {value}, not the stated value'
proof = steps(
    has_deriv_at_proof(d.body, d.var, d.point, value, problem, proofs=PROOFS, function=d.function, at=d.at),
    tactic('exact hd.deriv.trans (by first | rfl | ring1 | (field_simp; ring1) | norm_num)'),
)
