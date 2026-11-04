# Exercise exr-dq-eight: alpha_k' = alpha_k (x) 1_2 and beta' = beta (x) sigma_3 satisfy the Dirac algebra in size eight; the
# commutant (matrices commuting with all four) has dimension 4, so the set is reducible: two equivalent copies of size four.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
eta = diagonal_matrix(SR, [1, -1, -1, -1])
beta = g[0]; alpha = [g[0] * g[k] for k in (1, 2, 3)]
A = [a.tensor_product(one) for a in alpha]; B = beta.tensor_product(s[2])
I8 = identity_matrix(SR, 8)
print("relations in size 8:", all(A[j] * A[k] + A[k] * A[j] == 2 * kronecker_delta(j, k) * I8 for j in range(3) for k in range(3)) and all(a * B + B * a == 0 for a in A) and B^2 == I8)
X = matrix(SR, 8, 8, var(" ".join("x%d_%d" % (i, j) for i in range(8) for j in range(8))))
eqs = [e for M in A + [B] for e in (X * M - M * X).list()]
sol = solve(eqs, X.list(), solution_dict=True)[0]
free = set(v for val in sol.values() for v in SR(val).variables())
print("dimension of the commutant:", len(free))
# the two blocks sigma_3 = +1, -1 carry (alpha, beta) and (alpha, -beta), and gamma_5 maps one to the other
print("gamma_5 commutes with alpha_k and anticommutes with beta:", all(g5 * a == a * g5 for a in alpha) and g5 * beta == -beta * g5)
C = [I4.tensor_product(one), I4.tensor_product(s[2]), g5.tensor_product(s[0]), g5.tensor_product(s[1])]
print("1(x)1, 1(x)sigma_3, gamma_5(x)sigma_1, gamma_5(x)sigma_2 lie in the commutant:", all(X_ * M == M * X_ for X_ in C for M in A + [B]))
