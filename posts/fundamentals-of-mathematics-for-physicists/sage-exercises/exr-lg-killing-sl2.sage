# Exercise exr-lg-killing-sl2: the Killing form of sl(2, R) in the basis (H, E, F): kappa(H, H) = 8, kappa(E, F) = 4,
# all others zero; it is non-degenerate (sl(2) is simple) but indefinite, of signature (2, 1): SL(2, R) is not compact.
H = matrix(QQ, [[1, 0], [0, -1]]); E = matrix(QQ, [[0, 1], [0, 0]]); F = matrix(QQ, [[0, 0], [1, 0]])
basis = [H, E, F]
def br(X, Y): return X * Y - Y * X
Bmat = matrix(QQ, [b.list() for b in basis]).transpose()
def ad(X): return matrix(QQ, [Bmat.solve_right(vector(QQ, br(X, b).list())) for b in basis]).transpose()
K = matrix(QQ, 3, 3, lambda i, j: (ad(basis[i]) * ad(basis[j])).trace())
print("Killing form ="); print(K)
print("equals 4 tr(XY):", K == matrix(QQ, 3, 3, lambda i, j: 4 * (basis[i] * basis[j]).trace()))
print("det =", K.det(), "; eigenvalues =", K.eigenvalues())
sl2 = lie_algebras.sl(QQ, 2, representation="matrix")
print("Sage's lie_algebras.sl(QQ, 2) gives the same matrix in its own basis"); print(sl2.killing_form_matrix())
