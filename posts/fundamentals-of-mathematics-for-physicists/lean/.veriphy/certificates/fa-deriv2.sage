# name: fa-deriv2
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value (derivative recipe), with the
#   closing step widened so that products of sines and cosines also close (ring1 directly, not only after simp)
# goal: deriv (fun x => f x) a = v, f a polynomial in sin, cos, exp and powers (Legendre functions, heat kernels)
# description: Sage differentiates; mathlib's derivative lemmas, composed, prove HasDerivAt at the point; the convert
#   goals close by ring1 or ring_nf; deriv follows.
PROOFS = {}
assert problem.derivatives, 'the goal has no derivative Veriphy translates; inspect it with #veriphy_translate'
d = problem.derivatives[0]
value = diff(d.body, d.var).subs({d.var: d.point})
assert bool(goal.expr.simplify_full() == 0), f'Sage computes the derivative as {value}, not the stated value'
step = has_deriv_at_proof(d.body, d.var, d.point, value, problem, proofs=PROOFS, function=d.function, at=d.at)
text = getattr(step, 'text', str(step))
text = text.replace('all_goals first | rfl |', 'all_goals first | rfl | ring1 | (ring_nf; done) | (field_simp; ring1) |', 1)
proof = steps(
    tactic(text),
    tactic('exact hd.deriv.trans (by first | rfl | ring1 | (field_simp; ring1) | norm_num)'),
)
