# Exercise exr-sr-lg-columns: the columns of B(0.6 x) R_z(pi/2) form an orthonormal frame for the Minkowski product.
eta = diagonal_matrix(QQ, [1, -1, -1, -1])
B = matrix(QQ, [[5/4, 3/4, 0, 0], [3/4, 5/4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
R = matrix(QQ, [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
Lam = B*R
print("Lambda ="); print(Lam)
print("Gram matrix of the columns:"); print(Lam.transpose()*eta*Lam)
print("Gram matrix of the rows:"); print(Lam*eta*Lam.transpose())
