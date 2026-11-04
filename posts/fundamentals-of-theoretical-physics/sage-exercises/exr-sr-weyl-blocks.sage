# Exercise exr-sr-weyl-blocks: exp(zeta S^{01}) and exp(theta S^{12}) in the Weyl representation; the blocks are
# (A^dagger)^{-1} on the upper (gamma_5 = -1) components and A on the lower ones, and the boost covers exp(zeta K1).
var('zeta theta', domain='real')
s0 = identity_matrix(SR, 2); s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
Z = zero_matrix(SR, 2)
g = [block_matrix([[Z, sm], [sb, Z]], subdivide=False) for sm, sb in zip([s0, s1, s2, s3], [s0, -s1, -s2, -s3])]
print("gamma_5 diagonal:", (I*g[0]*g[1]*g[2]*g[3]).diagonal())
S = lambda m, n: (g[m]*g[n] - g[n]*g[m])/4
Sb = (zeta*S(0, 1)).exp().simplify_full(); Sr = (theta*S(1, 2)).exp().simplify_full()
A = (zeta/2*s1).exp().simplify_full(); Ar = (-I*theta/2*s3).exp().simplify_full()
print("boost: upper block = (A^dag)^-1:", (Sb[0:2, 0:2] - A.conjugate_transpose().inverse()).simplify_full() == 0,
      "  lower block = A:", (Sb[2:4, 2:4] - A).simplify_full() == 0)
print("rotation: both blocks = exp(-i theta sigma_3/2):", (Sr[0:2, 0:2] - Ar).simplify_full() == 0 and (Sr[2:4, 2:4] - Ar).simplify_full() == 0)
K1 = matrix(SR, 4, 4); K1[0, 1] = K1[1, 0] = 1
L = (zeta*K1).exp().simplify_full()
print("S^-1 gamma^mu S = Lambda^mu_nu gamma^nu:", all((Sb.inverse()*g[m]*Sb - sum(L[m, n]*g[n] for n in range(4))).simplify_full() == 0 for m in range(4)))
