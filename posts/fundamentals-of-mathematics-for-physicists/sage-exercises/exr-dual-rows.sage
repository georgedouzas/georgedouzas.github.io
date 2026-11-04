# Exercise exr-dual-rows: linear functionals on Q^3 as rows; the sum and the scalar multiple of two functionals are the
# sum and the multiple of their rows, and the three coordinate functionals form a basis of the dual space.
alpha, beta = vector(QQ, [2, 1, 0]), vector(QQ, [1, -1, 3])
v = vector(QQ, [1, 2, 3])
print("alpha(v) =", alpha * v, ", beta(v) =", beta * v, ", (alpha + beta)(v) =", (alpha + beta) * v, ", (4 alpha)(v) =", (4 * alpha) * v)
print("alpha + beta as a row:", alpha + beta, "; 4 alpha:", 4 * alpha)
e = identity_matrix(QQ, 3).rows()
print("alpha = 2 e^1 + 1 e^2 + 0 e^3:", alpha == 2 * e[0] + 1 * e[1] + 0 * e[2], "; the coordinate functionals span the dual (Q^3)*:", (QQ ^ 3).span(e) == QQ ^ 3)
