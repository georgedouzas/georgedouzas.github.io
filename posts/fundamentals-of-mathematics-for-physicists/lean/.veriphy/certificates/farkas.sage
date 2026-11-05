# name: farkas
# technique: Farkas refutation (exact LP)
# goal: False, or a ≤ b / a < b, from linear hypotheses (=, ≤, <) over an ordered field
# description: Find exact multipliers making the hypotheses sum to a contradictory constant; norm_num refutes it.
steps_before = []
cons = list(hyps)
if goal_kind == 'rel' and goal.rel in ('le', 'lt'):
    negated = fresh_name(problem, 'hneg')
    steps_before = [negate_goal(negated)]
    cons.append(Hyp(negated, -goal.poly, 'lt' if goal.rel == 'le' else 'le', ''))
n = len(cons)
usable = [h.poly is not None and h.rel in ('eq', 'le', 'lt') and h.poly.degree() <= 1 for h in cons]
forms = [h.poly if ok else R(0) for h, ok in zip(cons, usable)]
rows = [([QQ(int(j == i)) for j in range(n)], 0) for i in range(n) if not usable[i]]
rows += [([f.monomial_coefficient(x) for f in forms], 0) for x in R.gens()]
kappa = [f.constant_coefficient() for f in forms]
nonneg = [i for i, h in enumerate(cons) if h.rel in ('le', 'lt')]
# Minimize only over the sign-constrained multipliers: the multipliers of equalities are free, and with more equalities
# than unknowns they move along the left kernel, so Σ wᵢ over all of them is unbounded and the LP reports no solution.
objective = [1 if i in nonneg else 0 for i in range(n)]
weights = solve_exact_lp(n, eqs=rows, les=[([-k for k in kappa], 1)], objective=objective, nonneg=nonneg)
if weights is None:
    strict = [1 if ok and h.rel == 'lt' else 0 for h, ok in zip(cons, usable)]
    weights = solve_exact_lp(n, eqs=rows + [(kappa, 0)], les=[([-s for s in strict], 1)], objective=objective, nonneg=nonneg)
assert weights is not None, 'the hypotheses are consistent: no Farkas certificate'
constant = sum(w * k for w, k in zip(weights, kappa))
uses_strict = any(w != 0 and h.rel == 'lt' for w, h in zip(weights, cons))
weighted = [(w, term(h.name)) for w, h in zip(weights, cons)]
proof = steps(*steps_before, refute(constant, strict=uses_strict, weighted=weighted, problem=problem))
