# Exercise exr-dq-covariance-current-boost: for the boost S = exp(w S^{01}) with w = log 3 (cosh w = 5/3, sinh w = 4/3), the
# current of S psi is Lambda applied to the current of psi, and Omega, N are invariant.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
eta = diagonal_matrix(SR, [1, -1, -1, -1])
psi = vector(SR, [1 + 2 * I, -1, 3 * I, 2 - I])
S = (sqrt(3) + 1 / sqrt(3)) / 2 * I4 + (sqrt(3) - 1 / sqrt(3)) / 2 * (g[0] * g[1])   # cosh(w/2) + gamma^0 gamma^1 sinh(w/2)
print("S = exp(w S^{01}) (check: S^-1 gamma^0 S = cosh w gamma^0 + sinh w gamma^1):", (S.inverse() * g[0] * S - (QQ(5) / 3 * g[0] + QQ(4) / 3 * g[1])).simplify_full() == 0)
bil = lambda f, G_: (f.conjugate() * g[0] * G_ * f).simplify_full()
j = [bil(psi, g[m]) for m in range(4)]; jp = [bil(S * psi, g[m]) for m in range(4)]
Lam = matrix(QQ, [[5/3, 4/3, 0, 0], [4/3, 5/3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
print("j =", j, "; j' =", jp)
print("j' = Lambda j:", vector(SR, jp) == Lam * vector(SR, j))
print("Omega, N unchanged:", bool(bil(S * psi, I4) == bil(psi, I4)) and bool(bil(S * psi, I * g5) == bil(psi, I * g5)), "; Omega =", bil(psi, I4), ", N =", bil(psi, I * g5))
