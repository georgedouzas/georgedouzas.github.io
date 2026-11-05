# name: la-trig-sin-cube
# technique: rewrite with a named mathlib lemma, checked in Sage first (transcendental_rewrite recipe)
# goal: Real.sin x ^ 3 = 3 / 4 * Real.sin x - 1 / 4 * Real.sin (3 * x)  (sin³ as a combination of sin φ and sin 3φ)
# description: Sage confirms the identity; Real.sin_three_mul rewrites sin 3x and ring closes.
REWRITES = [('Real.sin_three_mul', False, [])]
PROOFS = {}
CLOSE = 'ring'
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full().canonicalize_radical() == 0) or bool(e.simplify_exp().simplify_trig() == 0) \
    or bool(e.simplify_trig().expand_trig().simplify_full() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
blocks = [lemma_rewrite(lemma, problem, reverse=rev, side=side, proofs=PROOFS) for lemma, rev, side in REWRITES]
proof = steps(*blocks, tactic(CLOSE)) if CLOSE else steps(*blocks)
