# Exercise exr-dq-helicity-eigen: for m = 1, p = (2, 2, 1) (E = sqrt 10, |p| = 3), the spinor u = (sqrt(E - 3) xi, sqrt(E + 3) xi)
# with sigma.p-hat xi = xi solves the Dirac equation and has helicity +1.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
p = vector(QQ, [2, 2, 1]); E = sqrt(10); m = 1
nsig = sum(p[k] * s[k] for k in range(3)) / 3
xi = vector(SR, [4 / 3, 2 / 3 + 2 * I / 3])
print("sigma.p-hat xi = xi:", (nsig * xi - xi).simplify_full() == 0)
u = vector(SR, list(sqrt(E - 3) * xi) + list(sqrt(E + 3) * xi))
pslash = E * g[0] - sum(p[k] * g[k + 1] for k in range(3))
print("(pslash - m) u = 0:", ((pslash - m * I4) * u).simplify_full() == 0)
Sig = [blk(sk, Z, Z, sk) for sk in s]
h = sum(p[k] * Sig[k] for k in range(3)) / 3
print("h(p) u = u:", (h * u - u).simplify_full() == 0)
