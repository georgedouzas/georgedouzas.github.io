# Exercise exr-qm-projection-degenerate: projection onto the degenerate eigenspace of A and the state after the outcome.
A = matrix(QQ, [[2, 1, 0], [1, 2, 0], [0, 0, 3]])
print("eigenvalues:", A.eigenvalues())
V3 = (A - 3).right_kernel()
B = matrix(SR, V3.basis()).transpose()
P3 = B * (B.transpose() * B).inverse() * B.transpose()
print("P_3 ="); print(P3)
print("P_3^2 = P_3, P_3^T = P_3:", P3 ** 2 == P3 and P3.transpose() == P3)
psi = vector(SR, [1, 2, 2]) / 3
Pp = P3 * psi
print("P_3 psi =", Pp, "; probability =", Pp * Pp, "; state after =", Pp / Pp.norm())
