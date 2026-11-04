# Exercise exr-kernel-image: kernel, image and rank of a linear map given by a matrix (Sage's matrix methods).
A = matrix(QQ, [[1, 2, 3], [2, 4, 6]])
print("A =", A.rows())
print("kernel (solutions of A x = 0):", A.right_kernel())
print("image (column space):", A.column_space())
print("rank + nullity =", A.rank(), "+", A.right_nullity(), "=", A.rank() + A.right_nullity(), "= number of columns", A.ncols())
print("A (1, 1, -1) =", A * vector(QQ, [1, 1, -1]))
