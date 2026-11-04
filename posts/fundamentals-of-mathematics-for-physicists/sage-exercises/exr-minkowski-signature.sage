# Exercise exr-minkowski-signature: bilinear forms as matrices; symmetry, non-degeneracy and signature with Sage's
# QuadraticForm (the form Q(v) = B(v, v) of the symmetric matrix B is QuadraticForm(QQ, 2*B)).
eta = diagonal_matrix(QQ, [1, -1, -1, -1])
v, w = vector(QQ, [1, 1, 0, 0]), vector(QQ, [1, 0, 0, 0])
print("eta(v, v) for v = (1, 1, 0, 0):", v * eta * v, " (null) ; eta(w, w) for w = (1, 0, 0, 0):", w * eta * w, " (timelike)")
print("symmetric:", eta.is_symmetric(), "; non-degenerate (det != 0):", eta.det() != 0)
print("signature (positive, negative, zero):", QuadraticForm(QQ, 2 * eta).signature_vector(), " (one time direction, three space directions)")
S = matrix(QQ, [[1, 2], [2, 4]])
print("the form with matrix [[1,2],[2,4]] is degenerate: det =", S.det(), "; kernel:", S.right_kernel().basis())
E = identity_matrix(QQ, 3)
print("the Euclidean inner product on R^3 has signature", QuadraticForm(QQ, 2 * E).signature_vector(), "(positive definite:", E.is_positive_definite(), ")")
