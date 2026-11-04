# Exercise exr-dq-parity-bilinears-sage: for psi' = gamma^0 psi, the bilinears of psi' against those of psi (scalar, pseudoscalar,
# vector components, axial-vector components).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
a = var("a0 a1 a2 a3", domain="real"); b = var("b0 b1 b2 b3", domain="real")
psi = vector(SR, [a[i] + I * b[i] for i in range(4)]); psp = g[0] * psi
bar = lambda f, M: (f.conjugate() * g[0] * M * f).expand()
def sign(M):
    x, y = bar(psp, M), bar(psi, M)
    return 1 if bool((x - y).simplify_full() == 0) else (-1 if bool((x + y).simplify_full() == 0) else None)
print("Omega:", sign(I4), " N:", sign(I * g5))
print("vector j^mu:", [sign(g[m]) for m in range(4)])
print("axial a^mu:", [sign(g[m] * g5) for m in range(4)])
