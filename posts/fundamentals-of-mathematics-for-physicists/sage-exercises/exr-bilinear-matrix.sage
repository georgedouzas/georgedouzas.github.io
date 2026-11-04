# Exercise exr-bilinear-matrix: the components B_ij = B(e_i, e_j) of B(v, w) = v1 w1 + 2 v1 w2 - v2 w1 on Q^2, the
# value B(v, w) = v^T B w on two vectors, and the check that B is not symmetric.
B = matrix(QQ, [[1, 2], [-1, 0]])
v, w = vector(QQ, [1, 3]), vector(QQ, [2, -1])
print("B =", B.rows())
print("B(v, w) = v^T B w =", v * B * w, "; by the formula v1 w1 + 2 v1 w2 - v2 w1 =", v[0] * w[0] + 2 * v[0] * w[1] - v[1] * w[0])
print("B(w, v) =", w * B * v, "; symmetric:", B.is_symmetric())
print("components: B(e1, e1), B(e1, e2), B(e2, e1), B(e2, e2) =", [e_i * B * e_j for e_i in identity_matrix(QQ, 2).rows() for e_j in identity_matrix(QQ, 2).rows()])
