# Exercise exr-bilinear-change-basis: under a change of basis e'_j = A^i_j e_i the components of a bilinear form become
# B' = A^T B A; the dot product of Q^2 in the basis f = ((2, 1), (1, 2)) is the Gram matrix [[5, 4], [4, 5]].
A = matrix(QQ, [[2, 1], [1, 2]]).transpose()      # columns f1, f2
B = identity_matrix(QQ, 2)                         # the dot product
Bp = A.transpose() * B * A
print("B' = A^T B A =", Bp.rows(), "= Gram matrix of f: f1.f1, f1.f2, f2.f2 =", A.column(0) * A.column(0), A.column(0) * A.column(1), A.column(1) * A.column(1))
v = vector(QQ, [5, 4]); c = A.inverse() * v       # f-coordinates (2, 1)
print("|v|^2 in the standard basis:", v * B * v, "; with the f-coordinates", c, "and B':", c * Bp * c)
