# Exercise exr-vs-symmetric-matrices: the symmetric 3 x 3 matrices form a subspace of the 3 x 3 matrices (closed under
# sums and scalar multiples), of dimension 6; the rotation matrices do not form a subspace.
M = MatrixSpace(QQ, 3, 3)
A = matrix(QQ, [[1, 2, 0], [2, 3, 1], [0, 1, 4]]); B = matrix(QQ, [[0, 1, 1], [1, 0, 2], [1, 2, 5]])
print("A, B symmetric:", A.is_symmetric(), B.is_symmetric(), "; A + B and 3A symmetric:", (A + B).is_symmetric(), (3 * A).is_symmetric())
basis = [m for m in M.basis()]
sym = [m + m.transpose() for m in basis]
S = M.span(sym) if hasattr(M, "span") else None
# the subspace of symmetric matrices, as the span of E_ii and E_ij + E_ji
sym_space = span([vector(QQ, (m + m.transpose()).list()) for m in basis], QQ)
print("dimension of the symmetric 3 x 3 matrices:", sym_space.dimension(), "= 3 * 4 / 2")
R1 = matrix(QQ, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]); R2 = matrix(QQ, [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
print("R1, R2 rotations (R^T R = 1, det 1):", R1.transpose() * R1 == 1 and R1.det() == 1, R2.transpose() * R2 == 1 and R2.det() == 1)
print("R1 + R2 is a rotation:", (R1 + R2).transpose() * (R1 + R2) == 1, " (so SO(3) is not a subspace)")
