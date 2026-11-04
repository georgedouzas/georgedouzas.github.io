# Exercise exr-sth-sage-decomposition: the spectral decomposition of [[2, 1, 0], [1, 2, 1], [0, 1, 2]]: the orthogonal
# matrix U of normalised eigenvectors, A = U D U^T, A = sum lambda_k |e_k><e_k| and sum |e_k><e_k| = 1.
A = matrix(SR, [[2, 1, 0], [1, 2, 1], [0, 1, 2]])
D, V = A.eigenmatrix_right()
U = matrix(SR, [c / c.norm() for c in V.columns()]).transpose().simplify_full()
print("D = diag", D.diagonal()); print("U ="); print(U)
print("U^T U = 1:", bool((U.transpose() * U).simplify_full() == identity_matrix(3)))
print("A = U D U^T:", bool((U * D * U.transpose()).simplify_full() == A))
Ps = [(U.column(k).column() * U.column(k).column().transpose()).simplify_full() for k in range(3)]
print("A = sum lambda_k P_k:", bool((sum(D[k, k] * Ps[k] for k in range(3))).simplify_full() == A), "; sum P_k = 1:", bool(sum(Ps).simplify_full() == identity_matrix(3)))
print("P_k P_l = delta_kl P_k:", all(bool((Ps[k] * Ps[l]).simplify_full() == (Ps[k] if k == l else 0)) for k in range(3) for l in range(3)))
