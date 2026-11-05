# name: ra-deriv-log
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value
# goal: deriv log x = 1/x (x > 0)
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

# the emitted closing lacks ring1/field_simp for the derivative-value goal (its `simp only [Pi.pow_apply]` makes no
# progress there) and Pi.pow_apply for the function goal when the term applies `(fun x => x) ^ n` pointwise
proof = tactic(str(proof).replace('all_goals first | rfl | ', 'all_goals first | rfl | ring1 | (field_simp; ring1) | (field_simp; ring_nf; done) | ')
    .replace('(funext y; simp only [Pi.add_apply]; ring_nf; done)', '(funext y; simp only [Pi.add_apply, Pi.pow_apply]; ring_nf; done) | (funext y; simp; ring_nf; done)'))
