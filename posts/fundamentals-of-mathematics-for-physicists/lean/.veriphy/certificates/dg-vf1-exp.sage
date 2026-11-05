# name: dg-vf1-exp
# technique: rewrite with named mathlib lemmas from the translation tables, checked in Sage first
# goal: exp t * exp s = exp (t + s), the group law of the dilation flow (differential-geometry chapter)
# description: Sage confirms the identity, then a rewrite the tables offer (core, packs, project) closes it.
# EDIT: the rewrites, as (lemma, right-to-left, side conditions); left empty, every table candidate is tried
REWRITES = []
PROOFS = {}  # EDIT: tactics proving side conditions no hypothesis states, by their Lean text
CLOSE = None  # EDIT: a tactic closing what remains after the rewrites, or None (ring1 or norm_num after a table rewrite)

e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full().canonicalize_radical() == 0) or bool(e.simplify_exp().simplify_trig() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
if REWRITES:
    blocks = [lemma_rewrite(lemma, problem, reverse=rev, side=side, proofs=PROOFS) for lemma, rev, side in REWRITES]
    proof = steps(*blocks, tactic(CLOSE)) if CLOSE else steps(*blocks)
else:
    # every rewrite the tables offer, tried in order: the replacement keeps the one that closed the goal
    candidates = rewrite_candidates(problem)
    assert candidates, (f'no rewrite in the tables applies to {goal.text}; list one in REWRITES, '
                        'or add a rewrite entry to .veriphy/translations/')
    close = tactic(CLOSE or 'all_goals first | ring1 | (norm_num; done)')
    proof = first_of(*(steps(c.emit(problem, proofs=PROOFS), close) for c in candidates))
