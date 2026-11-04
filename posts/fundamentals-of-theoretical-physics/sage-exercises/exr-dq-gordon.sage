# Exercise exr-dq-gordon: the current bar(u) gamma^mu u of the spinors u(p) for momentum p = m sinh w along z equals 2 p^mu for
# both spin states (E = m cosh w).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
eta = diagonal_matrix(SR, [1, -1, -1, -1])
w = var("w", domain="real"); m = var("m", domain="positive")
ups = {"up": sqrt(m) * vector(SR, [exp(-w / 2), 0, exp(w / 2), 0]), "down": sqrt(m) * vector(SR, [0, exp(w / 2), 0, exp(-w / 2)])}
p = [m * (exp(w) + exp(-w)) / 2, 0, 0, m * (exp(w) - exp(-w)) / 2]
for name, u in ups.items():
    cur = [(u * g[0] * g[mu] * u).simplify_full() for mu in range(4)]   # u is real for real w
    print("spin", name, ": bar(u) gamma^mu u =", cur, "; equals 2 p^mu:", all(bool((c - 2 * pm).simplify_full() == 0) for c, pm in zip(cur, p)))
