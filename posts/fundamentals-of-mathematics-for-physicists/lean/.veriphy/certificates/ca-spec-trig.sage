# name: ca-spec-trig
# technique: rewrite with named mathlib lemmas from the translation tables, checked in Sage first (transcendental_rewrite)
# goal: a trigonometric identity such as 1 - sin θ ^ 2 = cos θ ^ 2
# description: Sage confirms the identity, then a rewrite the tables offer (or Real.cos_sq') closes it.
REWRITES = [('Real.cos_sq\'', False, [])]
PROOFS = {}
CLOSE = None
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full() == 0) or bool(e.simplify_trig() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
blocks = [lemma_rewrite(lemma, problem, reverse=rev, side=side, proofs=PROOFS) for lemma, rev, side in REWRITES]
proof = steps(*blocks)
