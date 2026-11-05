# name: ca-spec-derivative-rational
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value (derivative recipe)
# goal: deriv (fun x => x / (1 + x)) x = 1 / (1 + x) ^ 2 with the hypothesis 1 + x ≠ 0
# description: Sage differentiates; mathlib's derivative lemmas, composed, prove HasDerivAt at the point; deriv follows.
PROOFS = {'x + 1 ≠ 0': 'rwa [add_comm]'}
assert problem.derivatives, 'the goal has no derivative Veriphy translates; inspect it with #veriphy_translate'
d = problem.derivatives[0]
value = diff(d.body, d.var).subs({d.var: d.point})
assert bool(goal.expr.simplify_full() == 0), f'Sage computes the derivative as {value}, not the stated value'
proof = steps(
    has_deriv_at_proof(d.body, d.var, d.point, value, problem, proofs=PROOFS, function=d.function, at=d.at),
    tactic('exact hd.deriv.trans (by first | rfl | ring1 | (field_simp; ring1) | norm_num)'),
)
