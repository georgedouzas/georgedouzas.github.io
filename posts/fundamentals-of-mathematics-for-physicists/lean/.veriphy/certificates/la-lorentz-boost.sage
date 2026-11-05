# name: la-lorentz-boost
# technique: matrix computation read from the goal (matrix_from_goal recipe) with a hyperbolic side identity, with the exercise as oracle
# goal: Λᵀ η Λ = η for a boost Λ with cosh u, sinh u entries (explicit !![…] literals)
# description: Sage evaluates lhs - rhs from the goal and finds, for every entry, the constant k with
#   entry = k (cosh² u - sinh² u - 1); Lean proves each entry with simp, then ring1 or
#   linear_combination k * Real.cosh_sq_sub_sinh_sq u (the multipliers Sage found).
e = goal.expr
# oracle (sage/exercises/exr-lorentz-boost.sage)
u_ = var('u_')
L = matrix(SR, [[cosh(u_), sinh(u_), 0, 0], [sinh(u_), cosh(u_), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
eta_ = diagonal_matrix(SR, [1, -1, -1, -1])
assert (L.transpose() * eta_ * L - eta_).simplify_full() == 0
print('oracle: Lambda^T eta Lambda = eta')
assert e is not None and hasattr(e, 'nrows'), 'the goal must be a matrix identity with a symbolic form'
vs = list(e.variables())
assert len(vs) == 1, f'one rapidity variable expected, got {vs}'
u = vs[0]
c, s = SR.var('c_'), SR.var('s_')
D = e.subs({cosh(u): c, sinh(u): s}).expand()
rel = c**2 - s**2 - 1
ks = set()
for i in range(D.nrows()):
    for j in range(D.ncols()):
        q = (D[i, j] / rel).simplify_full()
        assert q.is_constant(), f'entry ({i}, {j}) is not a multiple of cosh² - sinh² - 1: {D[i, j]}'
        ks.add(QQ(q))
name = atoms[str(u)]
alts = ['ring1'] + [f'linear_combination ({k}) * Real.cosh_sq_sub_sinh_sq {name}' for k in sorted(ks) if k != 0]
proof = tactic('ext i j\nfin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]\n'
               'all_goals first | ' + ' | '.join(alts))
