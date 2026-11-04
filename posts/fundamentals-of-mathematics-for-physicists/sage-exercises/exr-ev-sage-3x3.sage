# Exercise exr-ev-sage-3x3: eigenvalues and eigenvectors of the tridiagonal matrix [[2, 1, 0], [1, 2, 1], [0, 1, 2]], the
# orthogonality of the eigenvectors, and the trace and determinant as sum and product of the eigenvalues.
A = matrix(SR, [[2, 1, 0], [1, 2, 1], [0, 1, 2]])
D, V = A.eigenmatrix_right()
print("eigenvalues:", D.diagonal()); print("eigenvectors (columns):"); print(V)
print("A V = V D:", bool((A * V - V * D).simplify_full() == 0))
print("Gram matrix of the eigenvectors (diagonal = orthogonal):"); print((V.transpose() * V).simplify_full())
print("sum of eigenvalues =", sum(D.diagonal()).simplify_full(), "= tr A =", A.trace(), "; product =", prod(D.diagonal()).expand(), "= det A =", A.det())
print("characteristic polynomial:", A.change_ring(QQ).charpoly().factor())
