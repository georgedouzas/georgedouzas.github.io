# Exercise exr-dq-dirac-split: psi_1 = (psi + psi^c)/sqrt 2 and psi_2 = (psi - psi^c)/(i sqrt 2) are Majorana and (psi_1 + i psi_2)/sqrt 2 = psi.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
a = var("a0 a1 a2 a3", domain="real"); b = var("b0 b1 b2 b3", domain="real")
psi = vector(SR, [a[i] + I * b[i] for i in range(4)])
cc = lambda f: I * g[2] * f.conjugate()
p1 = (psi + cc(psi)) / sqrt(2); p2 = (psi - cc(psi)) / (I * sqrt(2))
print("psi_1^c = psi_1:", (cc(p1) - p1).simplify_full() == 0, "; psi_2^c = psi_2:", (cc(p2) - p2).simplify_full() == 0)
print("(psi_1 + i psi_2)/sqrt 2 = psi:", ((p1 + I * p2) / sqrt(2) - psi).simplify_full() == 0)
