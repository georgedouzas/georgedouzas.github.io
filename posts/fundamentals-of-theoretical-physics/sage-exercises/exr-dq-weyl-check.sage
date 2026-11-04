# Exercise exr-dq-weyl-check: the matrices alpha_k = gamma^0 gamma^k and beta = gamma^0 of the Weyl representation satisfy
# {alpha_j, alpha_k} = 2 delta_jk, {alpha_k, beta} = 0, beta^2 = 1; eigenvalues of H = alpha.p + beta m for p = (2, 3, 6), m = 24.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
eta = diagonal_matrix(SR, [1, -1, -1, -1])
beta = g[0]; alpha = [g[0] * g[k] for k in (1, 2, 3)]
print("alpha_k =", [a.diagonal() for a in alpha], "(diagonal blocks -sigma_k, sigma_k)")
print("{alpha_j, alpha_k} = 2 delta_jk:", all(alpha[j] * alpha[k] + alpha[k] * alpha[j] == 2 * kronecker_delta(j, k) * I4 for j in range(3) for k in range(3)))
print("{alpha_k, beta} = 0 and beta^2 = 1:", all(a * beta + beta * a == 0 for a in alpha) and beta^2 == I4)
H = 2 * alpha[0] + 3 * alpha[1] + 6 * alpha[2] + 24 * beta
print("eigenvalues of H:", sorted(H.eigenvalues()))
