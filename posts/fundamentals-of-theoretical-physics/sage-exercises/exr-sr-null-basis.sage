# Exercise exr-sr-null-basis: a basis of R^4 made of null vectors and its Gram matrix for the Minkowski form.
eta = diagonal_matrix([1, -1, -1, -1])
N = matrix(QQ, [[1, 1, 0, 0], [1, -1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])
print("norms:", [r*eta*r for r in N.rows()], "  det N =", N.det())
G = N*eta*N.transpose()
print("Gram matrix:"); print(G)
print("det G =", G.det(), "= (det N)^2 det eta =", N.det()^2*eta.det())
