# Exercise exr-dual-basis: the dual basis of f1 = (2, 1), f2 = (1, 2) in R^2, as the rows of the inverse of the matrix
# whose columns are f1, f2 (so that f^i(f_j) = delta^i_j), and the coordinates of a vector as the values of the dual basis.
B = matrix(QQ, [[2, 1], [1, 2]]).transpose()      # columns f1, f2
D = B.inverse()                                   # rows f^1, f^2
print("dual basis f^1, f^2 =", D.rows())
print("f^i(f_j) = delta^i_j:", D * B == identity_matrix(QQ, 2))
v = vector(QQ, [5, 4])
print("coordinates of v = (5, 4): f^1(v) =", D.row(0) * v, ", f^2(v) =", D.row(1) * v, " so v = 2 f1 + 1 f2:", 2 * B.column(0) + 1 * B.column(1) == v)
