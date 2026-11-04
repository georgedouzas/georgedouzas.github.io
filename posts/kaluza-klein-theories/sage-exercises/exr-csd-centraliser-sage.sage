# Exercise exr-csd-centraliser-sage: the centraliser of tau = (i/3) diag(1, 1, -2) in su(3) as a real linear
# system on the Gell-Mann coordinates, its basis, and its Lie algebra su(2) + u(1).
lam = [matrix(SR, [[0, 1, 0], [1, 0, 0], [0, 0, 0]]), matrix(SR, [[0, -I, 0], [I, 0, 0], [0, 0, 0]]), matrix(SR, [[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
       matrix(SR, [[0, 0, 1], [0, 0, 0], [1, 0, 0]]), matrix(SR, [[0, 0, -I], [0, 0, 0], [I, 0, 0]]), matrix(SR, [[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
       matrix(SR, [[0, 0, 0], [0, 0, -I], [0, I, 0]]), matrix(SR, [[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sqrt(3)]
tau = I / 3 * diagonal_matrix(SR, [1, 1, -2]); ad = lambda A, B: A * B - B * A
x = [var("x%d" % (k + 1), domain="real") for k in range(8)]
X = sum(I * x[k] * lam[k] for k in range(8))                     # a general element of su(3)
C = ad(tau, X)
eqs = [e for z in C.list() for e in (z.real_part(), z.imag_part()) if bool(z != 0)]
J = matrix(SR, [[diff(e, v) for v in x] for e in eqs])
ker = J.right_kernel().basis()
print("kernel of ad tau in Gell-Mann coordinates (x_1 .. x_8):", [list(v) for v in ker])
print("[PASS] the centraliser is spanned by i lambda_1, i lambda_2, i lambda_3, i lambda_8 (real dimension 4):",
      len(ker) == 4 and all(v[3] == 0 and v[4] == 0 and v[5] == 0 and v[6] == 0 for v in ker))
T = [I / 2 * lam[k] for k in (0, 1, 2)]; Y = I / 2 * lam[7]
eps = lambda a, b, c: sign(Permutation([a + 1, b + 1, c + 1])) if len({a, b, c}) == 3 else 0
ok = all(ad(T[a], T[b]) == sum(-eps(a, b, c) * T[c] for c in range(3)) for a in range(3) for b in range(3)) and all(ad(Y, t).is_zero() for t in T)
print("[PASS] [T_a, T_b] = -eps_abc T_c for T_a = (i/2) lambda_a, and i lambda_8 commutes with them: su(2) + u(1):", ok)
