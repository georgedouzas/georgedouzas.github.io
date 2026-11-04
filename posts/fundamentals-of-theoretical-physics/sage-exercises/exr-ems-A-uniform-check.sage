# Exercise exr-ems-A-uniform-check: two vector potentials of a uniform field.
x, y, z, B1, B2, B3, B = var('x y z B1 B2 B3 B')
X = vector([x, y, z]); Bv = vector([B1, B2, B3])
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)])
def div(F): return sum(diff(F[i], v) for i, v in enumerate([x, y, z]))
A1 = Bv.cross_product(X) / 2
print("curl(B x X/2) =", curl(A1).simplify_full(), "; div =", div(A1))
A1z = vector([-B * y / 2, B * x / 2, 0]); lam = -B * x * y / 2
print("A1 - grad(lambda) for B e_z, lambda = -B x y/2:", (A1z - vector([diff(lam, v) for v in [x, y, z]])).simplify_full())
