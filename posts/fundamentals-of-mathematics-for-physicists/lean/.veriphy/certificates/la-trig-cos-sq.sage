# name: la-trig-cos-sq
# technique: rewrite with a named mathlib lemma, checked in Sage first (transcendental_rewrite recipe)
# goal: Real.cos x ^ 2 = 1 / 2 + 1 / 2 * Real.cos (2 * x)  (cos² as a combination of 1 and cos 2φ)
# description: Sage confirms the identity; Real.cos_sq rewrites cos² and ring closes.
REWRITES = [('Real.cos_sq', False, [])]
PROOFS = {}
CLOSE = 'ring'
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full().canonicalize_radical() == 0) or bool(e.simplify_exp().simplify_trig() == 0) \
    or bool(e.simplify_trig().expand_trig().simplify_full() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
blocks = [lemma_rewrite(lemma, problem, reverse=rev, side=side, proofs=PROOFS) for lemma, rev, side in REWRITES]
proof = steps(*blocks, tactic(CLOSE)) if CLOSE else steps(*blocks)
