# Exercise exr-dq-c-properties: C = i gamma^2 gamma^0 in the Weyl representation; C gamma^mu C^-1 = -(gamma^mu)^T, C^T = -C, C^dag = C^-1,
# (psi^c)^c = psi, and for a commuting spinor bar(psi^c) psi^c = -bar(psi) psi, bar(psi^c) gamma^mu psi^c = bar(psi) gamma^mu psi.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
C = I * g[2] * g[0]
print("C gamma^mu C^-1 = -(gamma^mu)^T:", all(C * g[k] * C.inverse() == -g[k].transpose() for k in range(4)), "; C^T = -C:", C.transpose() == -C, "; C^dag = C^-1:", C.conjugate_transpose() == C.inverse())
a = var("a0 a1 a2 a3", domain="real"); b = var("b0 b1 b2 b3", domain="real")
psi = vector(SR, [a[i] + I * b[i] for i in range(4)])
cc = lambda f: I * g[2] * f.conjugate()
print("psi^c = C bar(psi)^T:", (cc(psi) - C * (psi.conjugate() * g[0])).simplify_full() == 0)
print("(psi^c)^c = psi:", (cc(cc(psi)) - psi).simplify_full() == 0)
bar = lambda f, M: (f.conjugate() * g[0] * M * f).expand()
print("bar(psi^c) psi^c = -bar(psi) psi:", bool((bar(cc(psi), I4) + bar(psi, I4)).simplify_full() == 0))
print("bar(psi^c) gamma^mu psi^c = bar(psi) gamma^mu psi:", all(bool((bar(cc(psi), g[k]) - bar(psi, g[k])).simplify_full() == 0) for k in range(4)))
