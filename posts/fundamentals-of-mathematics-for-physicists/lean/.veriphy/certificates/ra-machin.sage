# name: ra-machin
# technique: rewrite with named mathlib lemmas, checked in Sage first
# goal: Machin's formula 4 arctan(1/5) - arctan(1/239) = π/4
# description: Sage confirms the identity by 256-bit intervals; the rewrite is mathlib's Machin lemma.
REWRITES = [('Real.four_mul_arctan_inv_5_sub_arctan_inv_239', False, [])]  # the special values used, as (lemma, right-to-left, side conditions)
PROOFS = {}
CLOSE = None
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full() == 0) or bool(e.simplify_trig().simplify_full() == 0) or \
    RealIntervalField(256)(e).contains_zero(), f'Sage cannot confirm the identity: lhs - rhs = {e.simplify_full()}'
blocks = [lemma_rewrite(lemma, problem, reverse=rev, side=side, proofs=PROOFS) for lemma, rev, side in REWRITES]
proof = steps(*blocks, tactic(CLOSE)) if CLOSE else steps(*blocks)
