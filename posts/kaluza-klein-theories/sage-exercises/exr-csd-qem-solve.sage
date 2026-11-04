# Exercise exr-csd-qem-solve: the unbroken subalgebra at the vacuum Phi = (0, 1): the real elements
# X = c_1 T_1 + c_2 T_2 + c_3 T_3 + c_4 tau of h with [X, phi_1] = [X, phi_2] = 0.
c = var("c1 c2 c3 c4", domain="real")
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
ad = lambda A, B: A * B - B * A
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
def up(m): M = zero_matrix(SR, 3, 3); M[0:2, 0:2] = m; return M
T = [I / 2 * up(m) for m in s]; tau = I / 3 * diagonal_matrix(SR, [1, 1, -2])
X = c[0] * T[0] + c[1] * T[1] + c[2] * T[2] + c[3] * tau
Pp = E(1, 2); Pm = -E(2, 1); f1 = (Pp + Pm) / 2; f2 = I * (Pp - Pm) / 2
eqs = [e == 0 for M in (ad(X, f1), ad(X, f2)) for z in M.list() for e in (z.real_part(), z.imag_part()) if bool(e != 0)]
sol = solve(eqs, list(c), solution_dict=True)
print("solutions:", sol)
S = sol[0]; Xs = X.subs(S).simplify_full()
print("unbroken generator:", Xs)
Qem = I / 3 * diagonal_matrix(SR, [2, -1, -1])
print("[PASS] c_1 = c_2 = 0 and c_3 = 2 c_4, i.e. X = c_4 (2 T_3 + tau) = 2 c_4 Q_em: one unbroken generator:",
      bool(S[c[0]] == 0) and bool(S[c[1]] == 0) and (Xs - (Xs[0, 0] / Qem[0, 0]) * Qem).simplify_full().is_zero())
