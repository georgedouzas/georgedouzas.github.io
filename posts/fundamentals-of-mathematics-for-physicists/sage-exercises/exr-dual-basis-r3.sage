# Exercise exr-dual-basis-r3: the dual basis of b1 = (1, 1, 0), b2 = (0, 1, 1), b3 = (1, 0, 1) is the set of rows of the
# inverse of the matrix with columns b_j; it reads off the coordinates of (2, 3, 4) in the basis b.
B = matrix(QQ, [[1, 1, 0], [0, 1, 1], [1, 0, 1]]).transpose()
D = B.inverse()
print("dual basis rows b^1, b^2, b^3:", D.rows())
print("b^i(b_j) = delta^i_j:", D * B == 1)
w = vector(QQ, [2, 3, 4])
print("coordinates of (2, 3, 4) = (b^1(w), b^2(w), b^3(w)) =", D * w)
