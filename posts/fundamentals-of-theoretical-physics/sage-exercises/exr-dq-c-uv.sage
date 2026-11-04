# Exercise exr-dq-c-uv: i gamma^2 (u^s)^* = v^s with eta^s = i sigma_2 (xi^s)^* for p = sinh(w) along z (m = 1).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
w = var("w", domain="real")
ps = [exp(-w), exp(w)]   # p.sigma = diag(E - p, E + p) = diag(e^-w, e^w)
sq = diagonal_matrix(SR, [exp(-w / 2), exp(w / 2)]); sqb = diagonal_matrix(SR, [exp(w / 2), exp(-w / 2)])   # sqrt(p.sigma), sqrt(p.sigma-bar)
for xi in (vector(SR, [1, 0]), vector(SR, [0, 1])):
    u = vector(SR, list(sq * xi) + list(sqb * xi))
    eta = I * s[1] * xi.conjugate()
    v = vector(SR, list(sq * eta) + list(-sqb * eta))
    uc = I * g[2] * u.conjugate()
    print("xi =", xi, ": eta = i sigma_2 xi^* =", eta, "; i gamma^2 u^* =", [c.simplify_full() for c in uc], "; equals v:", (uc - v).simplify_full() == 0)
