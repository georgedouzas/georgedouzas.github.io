# Exercise exr-dq-u-dirac-rep: U maps the Weyl representation to the Dirac one; U u(p) = (sqrt(E + m) xi, sigma.p xi/sqrt(E + m))
# for spin up and momentum p = m sinh w along z (E = m cosh w).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
U = blk(one, one, -one, one) / sqrt(2)
gD = [U * x * U.transpose() for x in g]
print("U gamma^0 U^T = diag(1, 1, -1, -1):", gD[0] == diagonal_matrix(SR, [1, 1, -1, -1]), "; U gamma^k U^T = gamma^k:", all(gD[k] == g[k] for k in (1, 2, 3)))
w = var("w", domain="real"); m = var("m", domain="positive")
E = m * (exp(w) + exp(-w)) / 2; p = m * (exp(w) - exp(-w)) / 2
uW = sqrt(m) * vector(SR, [exp(-w / 2), 0, exp(w / 2), 0])
uD = U * uW
print("U u =", [c.simplify_full() for c in uD])
z = lambda e: bool(SR(e).expand().simplify_full() == 0)
print("upper^2 = E + m:", z(uD[0]^2 - (E + m)), "; lower^2 = p^2/(E + m):", z(uD[2]^2 * (E + m) - p^2), "; signs positive for w > 0:", bool(uD[2].subs(w=1, m=1) > 0))
