# Exercise exr-dq-t-sage: T = i gamma^1 gamma^3 satisfies T gamma^0* T^-1 = gamma^0, T gamma^k* T^-1 = -gamma^k, T T^* = -1, [T, gamma_5] = 0,
# and T Sigma_k^* T^-1 = -Sigma_k.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
T = I * g[1] * g[3]
print("T =", T.rows())
print("T gamma^0* T^-1 = gamma^0:", T * g[0].conjugate() * T.inverse() == g[0], "; T gamma^k* T^-1 = -gamma^k:", all(T * g[k].conjugate() * T.inverse() == -g[k] for k in (1, 2, 3)))
print("T T^* = -1:", T * T.conjugate() == -I4, "; T gamma_5 = gamma_5 T:", T * g5 == g5 * T)
Sig = [blk(sk, Z, Z, sk) for sk in s]
print("T Sigma_k^* T^-1 = -Sigma_k:", all(T * Sk.conjugate() * T.inverse() == -Sk for Sk in Sig))
