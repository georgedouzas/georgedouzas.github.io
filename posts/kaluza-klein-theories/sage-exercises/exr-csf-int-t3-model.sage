# Exercise exr-csf-int-t3-model: charges in the T_3 model, tau' = (i/2) diag(1,-1,0); H = U(1)_3 x U(1)_8 with t the
# eigenvalue of diag(1,-1,0)/2 and y that of diag(1,1,-2). Fermion in the 3, scalar f in E_12.
T = diagonal_matrix(QQ, [QQ(1) / 2, -QQ(1) / 2, 0]); Y = diagonal_matrix(QQ, [1, 1, -2])
e = identity_matrix(QQ, 3).columns()
ch = lambda v: ((T * v).dot_product(v), (Y * v).dot_product(v))
E12 = matrix(QQ, 3, 3, {(0, 1): 1})
t_f = [(T * E12 - E12 * T)[0, 1], (Y * E12 - E12 * Y)[0, 1]]
print("tau' charges q of e1, e2, e3:", [T[i, i] for i in range(3)])
print("psi_R = e1 component:", ch(e[0]), " psi_L = e2 component:", ch(e[1]))
print("scalar f in E_12 has (t, y) =", tuple(t_f))
bil = (ch(e[0])[0] - ch(e[1])[0], ch(e[0])[1] - ch(e[1])[1])
print("bar psi_L psi_R has (t, y) =", bil, "; f^* bar psi_L psi_R invariant:", bil == tuple(t_f))
