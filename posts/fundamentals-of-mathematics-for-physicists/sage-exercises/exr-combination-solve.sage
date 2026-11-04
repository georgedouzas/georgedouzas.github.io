# Exercise exr-combination-solve: the coefficients that write w = (4, 7, 2) as a linear combination of v1 = (1, 2, 0),
# v2 = (0, 1, 1), v4 = (0, 0, 1): solve the linear system whose columns are v1, v2, v4.
v1, v2, v4 = vector(QQ, [1, 2, 0]), vector(QQ, [0, 1, 1]), vector(QQ, [0, 0, 1])
w = vector(QQ, [4, 7, 2])
C = matrix(QQ, [v1, v2, v4]).transpose()          # columns v1, v2, v4
lam = C.solve_right(w)
print("coefficients (lambda^1, lambda^2, lambda^4) =", lam)
print("check:", lam[0] * v1 + lam[1] * v2 + lam[2] * v4 == w)
