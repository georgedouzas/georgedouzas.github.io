# Exercise exr-fc-sage-exp: e^{iA} for A = [[2, 1], [1, 2]] from the spectral decomposition and from Sage's matrix
# exponential, its unitarity and eigenvalues, and the square root sqrt A = ((sqrt 3 + 1)/2) 1 + ((sqrt 3 - 1)/2) sigma_1.
A = matrix(SR, [[2, 1], [1, 2]]); s1 = matrix(SR, [[0, 1], [1, 0]])
D, V = A.eigenmatrix_right()
U = matrix(SR, [c / c.norm() for c in V.columns()]).transpose()
E1 = (U * diagonal_matrix([exp(I * d) for d in D.diagonal()]) * U.transpose()).apply_map(lambda e: e.rectform().simplify_full())
E2 = (I * A).exp().apply_map(lambda e: e.rectform().simplify_full())
print("e^{iA} from U diag(e^{i lambda}) U^T ="); print(E1)
print("agrees with the matrix exponential:", bool((E1 - E2).simplify_full() == 0))
print("unitary:", bool((E1 * E1.conjugate_transpose()).simplify_full() == identity_matrix(2)), "; det(e^{iA} - e^{3i}) and det(e^{iA} - e^{i}) vanish:", [abs((E1 - exp(I * lam) * identity_matrix(2)).det().n(100)) < 1e-25 for lam in [3, 1]])
t = var("t", domain="real")
Et = (I * t * A).exp().apply_map(lambda e: e.rectform().simplify_full())
Et2 = (exp(2 * I * t) * (cos(t) * identity_matrix(2) + I * sin(t) * s1)).apply_map(lambda e: e.rectform().trig_reduce().simplify_full())
print("e^{itA} = e^{2it}(cos t + i sin t sigma_1) for symbolic t:", bool((Et - Et2).apply_map(lambda e: e.trig_reduce().simplify_full()) == 0))
print("det e^{iA} = e^{4i} = e^{i tr A}:", abs(E1.det().n(100) - exp(4 * I).n(100)) < 1e-25)
sq = (sqrt(3) + 1) / 2 * identity_matrix(2) + (sqrt(3) - 1) / 2 * s1
print("(sqrt A)^2 = A:", bool((sq * sq).simplify_full() == A), "; sqrt A positive (eigenvalues):", sq.eigenvalues())
