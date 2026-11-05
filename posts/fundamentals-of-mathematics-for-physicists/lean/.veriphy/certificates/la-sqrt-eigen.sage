# name: la-sqrt-eigen
# technique: matrix computation read from the goal (matrix_from_goal recipe) with √2 entries
# goal: M * v = λ • v for an explicit !![…] matrix and column v with entries in ℚ(√2)
# description: Sage evaluates lhs - rhs from the goal and checks it is zero; Lean proves each entry by simp, then
#   linear_combination with the multiple of (√2)² = 2 that Sage reads off the entry.
e = goal.expr
assert e is not None and hasattr(e, 'nrows'), 'the goal must be a matrix identity with a symbolic form'
assert e.simplify_full() == 0, f'the identity is false: lhs - rhs = {e.simplify_full()}'
r = SR.var('r_')
mults = set()
for i in range(e.nrows()):
    for j in range(e.ncols()):
        q = e[i, j].subs({sqrt(2): r}).expand()
        k = q.coefficient(r, 2)
        assert (q - k * (r**2 - 2)).expand().simplify_full() == 0 or (q.subs({r: sqrt(2)}).simplify_full() == 0)
        mults.add(QQ(k))
tacs = ' | '.join(['ring1'] + [f'linear_combination ({k}) * Real.sq_sqrt (show (0:ℝ) ≤ 2 by norm_num)' for k in sorted(mults) if k != 0])
proof = tactic('ext i j\nfin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]\nall_goals first | ' + tacs)
