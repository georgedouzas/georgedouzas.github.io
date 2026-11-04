# Exercise exr-dq-majorana-condition-sage: psi_M = (chi, -i sigma_2 chi^*) satisfies i gamma^2 psi_M^* = psi_M; e^{i alpha} psi_M does only if sin(alpha) = 0.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
a, b, c, d, al = var("a b c d alpha", domain="real")
chi = vector(SR, [a + I * b, c + I * d])
psiM = vector(SR, list(chi) + list(-I * s[1] * chi.conjugate()))
print("psi_M =", list(psiM))
print("i gamma^2 psi_M^* = psi_M:", (I * g[2] * psiM.conjugate() - psiM).simplify_full() == 0)
rot = exp(I * al) * psiM
dif = (I * g[2] * rot.conjugate() - rot).simplify_full()
print("i gamma^2 (e^{i alpha} psi_M)^* - e^{i alpha} psi_M =", [x.simplify_full() for x in dif])
print("= -2 i sin(alpha) psi_M:", all(bool((x + 2 * I * sin(al) * y).expand() == 0) for x, y in zip(dif, psiM)))
