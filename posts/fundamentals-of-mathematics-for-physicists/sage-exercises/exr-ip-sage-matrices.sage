# Exercise exr-ip-sage-matrices: the trace inner product <A, B> = tr(A^dagger B) on 2 x 2 matrices, orthogonality of the
# Pauli matrices, the expansion of a Hermitian matrix in the Pauli basis, and Cauchy-Schwarz on two complex vectors.
s0 = identity_matrix(SR, 2)
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
ip = lambda A, B: (A.conjugate_transpose() * B).trace()
gram = matrix(SR, 4, 4, lambda a, b: ip([s0, s1, s2, s3][a], [s0, s1, s2, s3][b]))
print("Gram matrix of 1, sigma_1, sigma_2, sigma_3:"); print(gram)
M = matrix(SR, [[1, 2 - I], [2 + I, 3]])
c = [ip(s, M) / 2 for s in [s0, s1, s2, s3]]
print("coefficients c_0, c_1, c_2, c_3 =", c, "; reconstruction:", bool(sum(ck * sk for ck, sk in zip(c, [s0, s1, s2, s3])) == M))
u = vector(SR, [1, I, 2]); v = vector(SR, [1 - I, 3, I])
lhs = abs(sum(conjugate(a) * b for a, b in zip(u, v))); rhs = u.norm() * v.norm()
print("|<u, v>| =", lhs.simplify_full(), " ||u|| ||v|| =", rhs.simplify_full(), " Cauchy-Schwarz:", bool(lhs <= rhs))
