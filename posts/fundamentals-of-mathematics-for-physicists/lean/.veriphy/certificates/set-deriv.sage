# name: set-deriv
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value (derivative recipe)
# goal: deriv (fun x => f x) x = v at a symbolic point, f a polynomial, a quotient (its denominator nonzero by a
#   hypothesis) or a product with exp: x³ + x, x/(1 - x²), x e^{-x}
# description: Sage differentiates and checks the claimed value; mathlib's derivative lemmas, composed, prove HasDerivAt
#   at the point; deriv follows.
PROOFS = {}
assert problem.derivatives, 'the goal has no derivative Veriphy translates; inspect it with #veriphy_translate'
d = problem.derivatives[0]
value = diff(d.body, d.var).subs({d.var: d.point})
print('Sage: d/dx', d.body, '=', value.simplify_full())
e = goal.expr
assert bool(e.simplify_full() == 0) or bool(e.simplify_full().canonicalize_radical() == 0), \
    f'Sage computes the derivative as {value}, not the stated value'
proof = steps(
    has_deriv_at_proof(d.body, d.var, d.point, value, problem, proofs=PROOFS, function=d.function, at=d.at),
    tactic('exact hd.deriv.trans (by first | rfl | ring1 | (field_simp; ring1) | (field_simp; ring_nf; done) '
           '| (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub, ← Real.exp_add]; field_simp; ring_nf; done) '
           '| norm_num)'),
)
