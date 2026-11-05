# name: fa-trig-addition
# technique: rewrite with named mathlib lemmas, checked in Sage first (transcendental_rewrite recipe, REWRITES edited)
# goal: product-to-sum identities sin u cos v = (sin (u + v) + sin (u - v))/2 written with the sums u + v, u - v
# description: Sage confirms the identity; Real.sin_add and Real.sin_sub expand the right side and ring closes.
REWRITES = [('Real.sin_add', False, []), ('Real.sin_sub', False, [])]
PROOFS = {}
CLOSE = 'ring'
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full().canonicalize_radical() == 0) or bool(e.simplify_exp().simplify_trig() == 0) \
    or bool(e.trig_reduce().simplify_full() == 0) or bool(e.expand_trig().simplify_full() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
blocks = [lemma_rewrite(lemma, problem, reverse=rev, side=side, proofs=PROOFS) for lemma, rev, side in REWRITES]
proof = steps(*blocks, tactic(CLOSE)) if CLOSE else steps(*blocks)
