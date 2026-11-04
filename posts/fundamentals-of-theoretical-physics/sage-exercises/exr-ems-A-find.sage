# Exercise exr-ems-A-find: a vector potential for B = (B0/a)(0, 0, x) and the current that produces it.
x, y, z, B0, a = var('x y z B0 a')
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)])
B = vector([0, 0, B0 * x / a])
print("div B =", sum(diff(B[i], v) for i, v in enumerate([x, y, z])))
A = vector([0, B0 * x^2 / (2 * a), 0])
print("curl A =", curl(A), "; J = curl B =", curl(B))
