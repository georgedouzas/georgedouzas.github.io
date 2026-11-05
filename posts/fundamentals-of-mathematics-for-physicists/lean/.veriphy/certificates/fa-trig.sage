# name: fa-trig
# technique: rewrite with named mathlib lemmas from the translation tables, checked in Sage first (transcendental_rewrite recipe)
# goal: a trigonometric identity (power reduction, multiple angles) between sin and cos expressions
# description: Sage confirms the identity, then a rewrite the tables offer (power reduction, multiple angles) closes it.
REWRITES = []
PROOFS = {}
CLOSE = None
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
assert bool(e.simplify_full().canonicalize_radical() == 0) or bool(e.simplify_exp().simplify_trig() == 0) \
    or bool(e.trig_reduce().simplify_full() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs simplifies to {e.simplify_full()}'
if REWRITES:
    blocks = [lemma_rewrite(lemma, problem, reverse=rev, side=side, proofs=PROOFS) for lemma, rev, side in REWRITES]
    proof = steps(*blocks, tactic(CLOSE)) if CLOSE else steps(*blocks)
else:
    candidates = rewrite_candidates(problem)
    assert candidates, (f'no rewrite in the tables applies to {goal.text}; list one in REWRITES, '
                        'or add a rewrite entry to .veriphy/translations/')
    close = tactic(CLOSE or 'all_goals first | ring1 | (norm_num; done)')
    proof = first_of(*(steps(c.emit(problem, proofs=PROOFS), close) for c in candidates))
