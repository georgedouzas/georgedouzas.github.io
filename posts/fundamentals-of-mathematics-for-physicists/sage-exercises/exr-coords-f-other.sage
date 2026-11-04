# Exercise exr-coords-f-other: coordinates of (1, 5) and of (-1, 1) in the basis f1 = (2, 1), f2 = (1, 2), by solving
# the 2 x 2 system and by the inverse matrix.
f1, f2 = vector(QQ, [2, 1]), vector(QQ, [1, 2])
F = matrix(QQ, [f1, f2]).transpose()
for v in [vector(QQ, [1, 5]), vector(QQ, [-1, 1]), vector(QQ, [5, 4])]:
    c = F.inverse() * v
    print("v =", v, ": coordinates", c, "; check", c[0] * f1 + c[1] * f2 == v)
