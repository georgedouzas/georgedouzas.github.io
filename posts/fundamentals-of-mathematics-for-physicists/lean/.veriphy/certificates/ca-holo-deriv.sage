# name: ca-holo-deriv
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value (derivative recipe)
# goal: deriv (fun x => f x) a = v, f a polynomial, rational, exp/sin/cos expression in x with other variables as
#   parameters (the partial derivatives u_x, u_y, v_x, v_y of the parts of z², z³, eᶻ, 1/z, cos z, ...)
# description: Sage differentiates; mathlib's derivative lemmas, composed, prove HasDerivAt at the point; deriv follows.
#   Every non-vanishing side condition (a denominator x² + y²) is discharged from the hypothesis h : x ^ 2 + y ^ 2 ≠ 0.
PROOFS = {k: 'first | assumption | exact h | exact pow_ne_zero _ h | (convert h using 1; ring) | exact pow_ne_zero _ (by convert h using 1; ring) | positivity' for k in ['x ^ 2 + (y ^ 2) ≠ 0', 'x ^ 2 + y ^ 2 ≠ 0', '(x ^ 2) + y ^ 2 ≠ 0', '(x ^ 2 + (y ^ 2)) ^ 2 ≠ 0', '(x ^ 2 + y ^ 2) ^ 2 ≠ 0', '((x ^ 2) + y ^ 2) ^ 2 ≠ 0']}
assert problem.derivatives, 'the goal has no derivative Veriphy translates; inspect it with #veriphy_translate'
d = problem.derivatives[0]
value = diff(d.body, d.var).subs({d.var: d.point})
assert bool(goal.expr.simplify_full() == 0), f'Sage computes the derivative as {value}, not the stated value'
proof = steps(
    has_deriv_at_proof(d.body, d.var, d.point, value, problem, proofs=PROOFS, function=d.function, at=d.at),
    tactic('exact hd.deriv.trans (by first | rfl | ring1 | (field_simp; ring1) | norm_num)'),
)
