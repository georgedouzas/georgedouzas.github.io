# name: fa-deriv-pow
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value (derivative recipe), with the
#   nonzero conditions of the powers of the variable discharged from the hypothesis `ht : t ≠ 0` (PROOFS edited)
# goal: deriv (fun t => f t) t = v, f with quotients by powers t ^ k of the point (the bump function e^{-1/t} / t^k)
# description: Sage differentiates; mathlib's derivative lemmas, composed, prove HasDerivAt at the point, each
#   `t ^ k ≠ 0` by pow_ne_zero from `ht`; deriv follows.
PROOFS = {f't ^ {k} ≠ 0': f'exact pow_ne_zero {k} ht' for k in range(2, 9)}
assert problem.derivatives, 'the goal has no derivative Veriphy translates; inspect it with #veriphy_translate'
d = problem.derivatives[0]
value = diff(d.body, d.var).subs({d.var: d.point})
assert bool(goal.expr.simplify_full() == 0), f'Sage computes the derivative as {value}, not the stated value'
proof = steps(
    has_deriv_at_proof(d.body, d.var, d.point, value, problem, proofs=PROOFS, function=d.function, at=d.at),
    tactic('exact hd.deriv.trans (by first | rfl | ring1 | (field_simp; ring1) | norm_num)'),
)
