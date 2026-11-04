# Exercise exr-qz-fermion-mode: a fermionic mode as 2x2 matrices (anticommutator {b, b^dag} = 1, b^2 = 0, occupation
# 0 or 1) and two modes by the Jordan-Wigner construction b1 = b (x) 1, b2 = sigma_3 (x) b, which anticommute.
b = matrix(QQ, [[0, 1], [0, 0]]); bd = b.transpose(); one = identity_matrix(QQ, 2); s3 = diagonal_matrix(QQ, [1, -1])
ac = lambda A, B: A * B + B * A
print("{b, b^dag} = 1:", ac(b, bd) == one, "  b^2 = 0:", b * b == 0)
print("N = b^dag b =", (bd * b).list(), " eigenvalues", sorted((bd * b).eigenvalues()))
b1 = b.tensor_product(one); b2 = s3.tensor_product(b)
print("{b1, b2} = 0:", ac(b1, b2) == 0, "  {b1, b2^dag} = 0:", ac(b1, b2.transpose()) == 0)
print("{b2, b2^dag} = 1:", ac(b2, b2.transpose()) == identity_matrix(QQ, 4), "  b2^2 = 0:", b2 * b2 == 0)
naive = one.tensor_product(b)
print("without sigma_3 the modes commute instead: [b1, 1 (x) b] = 0:", b1 * naive - naive * b1 == 0, ", {b1, 1 (x) b} = 0:", ac(b1, naive) == 0)
