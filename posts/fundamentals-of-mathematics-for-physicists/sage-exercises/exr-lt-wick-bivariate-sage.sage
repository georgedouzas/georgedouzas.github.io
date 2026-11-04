# Exercise exr-lt-wick-bivariate-sage: two correlated Gaussians with precision matrix A = [[2, 1], [1, 3]]; the
# covariance C = A^-1 and the Wick formulas E X^2 Y^2 = Cxx Cyy + 2 Cxy^2, E X^3 Y = 3 Cxx Cxy by symbolic integration.
x, y = var("x y")
A = matrix(QQ, [[2, 1], [1, 3]])
C = A.inverse()
print("covariance matrix C = A^-1 =\n", C)
Q = (vector([x, y]) * A * vector([x, y])) / 2
Z = integrate(integrate(exp(-Q), y, -oo, oo), x, -oo, oo)
print("normalisation Z =", Z.simplify_full(), "= 2 pi / sqrt(det A):", bool((Z / (2 * pi / sqrt(A.det()))).canonicalize_radical() == 1))
E = lambda f: (integrate(integrate(f * exp(-Q), y, -oo, oo), x, -oo, oo) / Z).simplify_full()
print("E X^2 =", E(x ** 2), ", E Y^2 =", E(y ** 2), ", E XY =", E(x * y), "; equal to C:", bool(E(x ** 2) == C[0, 0]) and bool(E(y ** 2) == C[1, 1]) and bool(E(x * y) == C[0, 1]))
print("E X^2 Y^2 =", E(x ** 2 * y ** 2), "; Wick: Cxx Cyy + 2 Cxy^2 =", C[0, 0] * C[1, 1] + 2 * C[0, 1] ** 2, ":", bool(E(x ** 2 * y ** 2) == C[0, 0] * C[1, 1] + 2 * C[0, 1] ** 2))
print("E X^3 Y =", E(x ** 3 * y), "; Wick: 3 Cxx Cxy =", 3 * C[0, 0] * C[0, 1], ":", bool(E(x ** 3 * y) == 3 * C[0, 0] * C[0, 1]))
print("E X^4 =", E(x ** 4), "; Wick: 3 Cxx^2 =", 3 * C[0, 0] ** 2, ":", bool(E(x ** 4) == 3 * C[0, 0] ** 2))
