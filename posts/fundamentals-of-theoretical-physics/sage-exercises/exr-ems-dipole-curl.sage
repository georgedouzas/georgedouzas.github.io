# Exercise exr-ems-dipole-curl: the curl of m x X/(4 pi r^3) is the dipole field.
x, y, z, m1, m2, m3 = var('x y z m1 m2 m3')
X = vector([x, y, z]); m = vector([m1, m2, m3]); r = sqrt(x^2 + y^2 + z^2)
A = m.cross_product(X) / (4 * pi * r^3)
curlA = vector([diff(A[2], y) - diff(A[1], z), diff(A[0], z) - diff(A[2], x), diff(A[1], x) - diff(A[0], y)])
B = (3 * m.dot_product(X) * X / r^2 - m) / (4 * pi * r^3)
print("curl A - B =", (curlA - B).simplify_full())
print("div A =", sum(diff(A[i], v) for i, v in enumerate([x, y, z])).simplify_full())
