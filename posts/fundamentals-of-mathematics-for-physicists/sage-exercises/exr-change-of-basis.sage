# Exercise exr-change-of-basis: the change-of-basis matrix between f = ((2, 1), (1, 2)) and g = ((1, 1), (1, -1)): the
# coordinates of a vector in g are M times its coordinates in f, with M = G^-1 F (columns the g-coordinates of f1, f2).
F = matrix(QQ, [[2, 1], [1, 2]]).transpose(); G = matrix(QQ, [[1, 1], [1, -1]]).transpose()
M = G.inverse() * F
print("M = G^-1 F =", M.rows(), "; its columns are the g-coordinates of f1, f2")
v = vector(QQ, [5, 4])
cf, cg = F.inverse() * v, G.inverse() * v
print("coordinates of (5, 4): in f", cf, ", in g", cg, "; M cf == cg:", M * cf == cg)
