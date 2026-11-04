# Exercise exr-la-gram-f (integrative): the dot product of Q^2 as a bilinear form in the basis f = ((2, 1), (1, 2)): the
# Gram matrix G = A^T A, its signature, lowering the index of the vector with f-coordinates (2, 1), raising it back with
# G^-1, and the length of v computed in both bases.
A = matrix(QQ, [[2, 1], [1, 2]]).transpose()
G = A.transpose() * A
print("Gram matrix G_ij = f_i . f_j =", G.rows(), "; symmetric:", G.is_symmetric(), "; det =", G.det(), "; signature:", QuadraticForm(QQ, 2 * G).signature_vector())
c = vector(QQ, [2, 1])             # f-coordinates of v = (5, 4)
low = G * c
print("v^i = (2, 1) lowered: v_i = G_ij v^j =", low, "; raised again with G^-1:", G.inverse() * low)
v = A * c
print("|v|^2 = v_i v^i =", low * c, "= standard |v|^2 =", v * v)
