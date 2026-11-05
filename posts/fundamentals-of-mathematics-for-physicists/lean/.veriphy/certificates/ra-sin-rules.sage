# name: ra-sin-rules
# technique: rewrite with named mathlib lemmas, checked in Sage first
# goal: the trapezoidal and Simpson sums of sin on [0, π] with h = π/4
# description: Sage confirms the value; the special values of sin rewrite the sum and ring closes.
REWRITES = [('Real.sin_zero', False, []), ('Real.sin_pi', False, []), ('Real.sin_pi_div_two', False, []), ('Real.sin_pi_div_four', False, [])]  # the special values used, as (lemma, right-to-left, side conditions)
PROOFS = {}
CLOSE = 'ring'
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full() == 0) or bool(e.simplify_trig().simplify_full() == 0) or \
    RealIntervalField(256)(e).contains_zero(), f'Sage cannot confirm the identity: lhs - rhs = {e.simplify_full()}'
blocks = [lemma_rewrite(lemma, problem, reverse=rev, side=side, proofs=PROOFS) for lemma, rev, side in REWRITES]
proof = steps(*blocks, tactic(CLOSE)) if CLOSE else steps(*blocks)
