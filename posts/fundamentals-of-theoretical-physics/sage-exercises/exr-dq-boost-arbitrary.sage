# Exercise exr-dq-boost-arbitrary: for S = exp(w n_k S^{0k}) with n = (1, 2, 2)/3 and w = log 2, S^{-1} gamma^mu S equals
# Lambda^mu_nu gamma^nu for the vector boost along n (cosh w = 5/4, sinh w = 3/4).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
eta = diagonal_matrix(SR, [1, -1, -1, -1])
n = vector(QQ, [1, 2, 2]) / 3; ch, sh = QQ(5) / 4, QQ(3) / 4
# n_k S^{0k} = (1/2) n_k alpha_k squares to 1/4, so exp(w n.S) = cosh(w/2) + 2 n.S sinh(w/2), with cosh(w/2) = 3/(2 sqrt 2), sinh(w/2) = 1/(2 sqrt 2)
nS = sum(n[k] * g[0] * g[k + 1] for k in range(3)) / 2
print("(n.S^{0k})^2 = 1/4:", nS^2 == I4 / 4)
S = 3 / (2 * sqrt(2)) * I4 + 2 * nS / (2 * sqrt(2))
Lam = matrix(SR, 4, 4)
Lam[0, 0] = ch
for k in range(3):
    Lam[0, k + 1] = Lam[k + 1, 0] = n[k] * sh
    for l in range(3):
        Lam[k + 1, l + 1] = kronecker_delta(k, l) + (ch - 1) * n[k] * n[l]
print("Lambda is a Lorentz transformation:", (Lam.transpose() * eta * Lam - eta).simplify_full() == 0)
Si = S.inverse()
print("S^-1 gamma^mu S = Lambda^mu_nu gamma^nu for all mu:", all((Si * g[mu] * S - sum(Lam[mu, nu] * g[nu] for nu in range(4))).simplify_full() == 0 for mu in range(4)))
