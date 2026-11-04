# Exercise exr-csd-qem-other-vacuum: the vacuum Phi = (1, 0) (p = 1, q = 0) instead of (0, 1). Its unbroken generator
# Q' = (i/3) diag(-1, 2, -1), and the Weyl element w of SU(2) that maps one vacuum to the other and Q_em to Q'.
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
ad = lambda A, B: A * B - B * A
a, b = var("a b")
D = I * diagonal_matrix(SR, [a, b, -a - b])
cond = [e == 0 for e in ad(D, E(0, 2)).list() if bool(e != 0)]
print("[D, E_13] = 0 requires:", solve(cond, a))
Qp = I / 3 * diagonal_matrix(SR, [-1, 2, -1]); Qem = I / 3 * diagonal_matrix(SR, [2, -1, -1])
print("[PASS] the diagonal generators fixing E_13 are the multiples of Q' = (i/3) diag(-1, 2, -1):", solve(cond, a) == [a == -b / 2] and ad(Qp, E(0, 2)).is_zero())
w = matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])           # exp(-pi T_2) in the upper block: an element of SU(2) in H
print("[PASS] w = exp(-pi T_2) is in SU(3) and commutes with tau; w E_23 w^-1 = -E_13 and w Q_em w^-1 = Q':",
      w.det() == 1 and w * w.transpose() == identity_matrix(3) and ad(w, I / 3 * diagonal_matrix(SR, [1, 1, -2])).is_zero()
      and w * E(1, 2) * w.inverse() == -E(0, 2) and w * Qem * w.inverse() == Qp)
