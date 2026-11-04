# Exercise exr-signature-sage: signatures of constant metrics with QuadraticForm(...).signature_vector(): the Minkowski
# metric, the five-dimensional product metric with a circle, and three non-diagonal 2 x 2 matrices, one degenerate.
def signature(G):
    return QuadraticForm(QQ, 2 * G).signature_vector()      # (positive, negative, zero) counts of eigenvalues
eta = diagonal_matrix(QQ, [1, -1, -1, -1])
print("Minkowski eta = diag(1, -1, -1, -1): signature (p, q, z) =", signature(eta), "; det =", eta.det())
g5 = diagonal_matrix(QQ, [1, -1, -1, -1, -1])
print("five-dimensional diag(1, -1, -1, -1, -1): signature", signature(g5), "; det =", g5.det())
A = matrix(QQ, [[1, 2], [2, 1]]); B = matrix(QQ, [[2, 1], [1, 2]]); C = matrix(QQ, [[1, 2], [2, 4]])
for name, G in [("[[1, 2], [2, 1]]", A), ("[[2, 1], [1, 2]]", B), ("[[1, 2], [2, 4]]", C)]:
    print(name, ": eigenvalues", sorted(G.eigenvalues()), "; signature", signature(G), "; det", G.det(), "; non-degenerate:", G.det() != 0)
