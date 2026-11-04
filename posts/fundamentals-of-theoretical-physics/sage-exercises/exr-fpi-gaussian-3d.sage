# Exercise exr-fpi-gaussian-3d: the three-dimensional Gaussian integral with a source,
# int d^3x exp(-x.A.x/2 + J.x) = (2 pi)^{3/2} det(A)^{-1/2} exp(J.A^{-1}.J/2), A = [[2,1,0],[1,2,1],[0,1,2]].
x1, x2, x3, J1, J2, J3 = var("x1 x2 x3 J1 J2 J3")
A = matrix(QQ, [[2, 1, 0], [1, 2, 1], [0, 1, 2]]); X = vector([x1, x2, x3]); Jv = vector([J1, J2, J3])
print("det A =", A.det(), "  A^{-1} =", A.inverse().list())
I3 = integrate(integrate(integrate(exp(-X * A * X / 2 + Jv * X), x1, -oo, oo), x2, -oo, oo), x3, -oo, oo)
target = (2 * pi) ** (QQ(3) / 2) / sqrt(A.det()) * exp(Jv * A.inverse() * Jv / 2)
print("iterated integral equals the formula:", bool((I3 / target).simplify_full() == 1))
