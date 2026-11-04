# Exercise exr-emw-zones-check: the full fields of an oscillating dipole solve Maxwell's equations (time factor exp(-i w t), w = k).
x, y, z = var("x y z"); k, p0 = var("k p0", domain="positive")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)])
r = sqrt(x^2 + y^2 + z^2); n = vector(X) / r; p = vector([0, 0, p0])
ph = exp(I * k * r)
E = ph / (4 * pi) * (k^2 * n.cross_product(p).cross_product(n) / r + (3 * n * n.dot_product(p) - p) * (1 / r^3 - I * k / r^2))
B = k^2 * ph / (4 * pi) * n.cross_product(p) / r * (1 - 1 / (I * k * r))
print('div E =', div(E), '; div B =', div(B))
print('curl E - i k B =', (curl(E) - I * k * B).simplify_full())
print('curl B + i k E =', (curl(B) + I * k * E).simplify_full())
