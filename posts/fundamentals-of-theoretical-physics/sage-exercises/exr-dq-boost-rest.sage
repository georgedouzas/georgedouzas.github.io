# Exercise exr-dq-boost-rest: S = exp(w n_k S^{0k}) with n = (3, 0, 4)/5, w = log 2 maps u^s(0) = (xi, xi) (m = 1) to
# u^s(p) = (sqrt(p.sigma) xi, sqrt(p.sigma-bar) xi) with sqrt(p.sigma) = exp(-w n.sigma/2), p = sinh(w) n.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
n = vector(QQ, [3, 0, 4]) / 5
ns = sum(n[k] * s[k] for k in range(3))
c, sh = 3 / (2 * sqrt(2)), 1 / (2 * sqrt(2))      # cosh(w/2), sinh(w/2) for w = log 2
Sl = c * one - sh * ns; Sr = c * one + sh * ns       # exp(-w n.sigma/2), exp(+w n.sigma/2)
nS = sum(n[k] * g[0] * g[k + 1] for k in range(3)) / 2
S = c * I4 + 2 * sh * nS
print("S = diag(exp(-w n.sigma/2), exp(w n.sigma/2)):", S == blk(Sl, Z, Z, Sr))
ch, shw = QQ(5) / 4, QQ(3) / 4; p = shw * n
psig = ch * one - sum(p[k] * s[k] for k in range(3)); psigb = ch * one + sum(p[k] * s[k] for k in range(3))
print("Sl^2 = p.sigma and Sr^2 = p.sigma-bar:", (Sl^2 - psig).simplify_full() == 0, (Sr^2 - psigb).simplify_full() == 0)
pslash = ch * g[0] - sum(p[k] * g[k + 1] for k in range(3))
for xi in (vector(SR, [1, 0]), vector(SR, [0, 1])):
    u0 = vector(SR, list(xi) + list(xi)); u = S * u0
    print("xi =", xi, ": S u(0) =", [x.simplify_full() for x in u], "; (pslash - 1) S u(0) = 0:", ((pslash - I4) * u).simplify_full() == 0)
