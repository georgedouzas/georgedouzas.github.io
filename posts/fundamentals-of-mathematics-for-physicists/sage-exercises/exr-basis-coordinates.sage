# Exercise exr-basis-coordinates: (1, 1, 0), (0, 1, 1), (1, 0, 1) is a basis of Q^3 (determinant 2 != 0), and the
# coordinates of (2, 3, 4) in it are the solution of the linear system with these columns.
V = QQ ^ 3
b = [vector(QQ, [1, 1, 0]), vector(QQ, [0, 1, 1]), vector(QQ, [1, 0, 1])]
C = matrix(QQ, b).transpose()
print("determinant of the matrix of columns b1, b2, b3:", C.det(), "; basis of Q^3:", V.span(b) == V and V.linear_dependence(b) == [])
w = vector(QQ, [2, 3, 4])
coords = V.span_of_basis(b).coordinates(w)
print("coordinates of (2, 3, 4):", coords, "; check:", sum(c * v for c, v in zip(coords, b)) == w)
print("the same from the inverse matrix C^-1 w:", C.inverse() * w)
