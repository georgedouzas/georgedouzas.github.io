# Exercise exr-clm-curl-test: curls of three forces and potentials V with F = -grad V where the curl vanishes.
x, y, z, k = var("x y z k", domain="real")
X = vector([x, y, z]); r = sqrt(x**2 + y**2 + z**2)
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)])
F1 = vector([2 * x * y, x**2, 0]); F2 = vector([y, -x, 0]); F3 = -k * X / r**3
for name, F in [("F1", F1), ("F2", F2), ("F3", F3)]:
    print("curl", name, "=", curl(F).simplify_full())
V1 = -x**2 * y; V3 = -k / r
print("-grad V1 == F1:", bool(-V1.gradient([x, y, z]) == F1), "; -grad V3 == F3:", bool((-V3.gradient([x, y, z]) - F3).simplify_full() == 0))
s = var("s")
print("work of F2 around the unit circle:", integrate(F2.subs(x=cos(s), y=sin(s)) * vector([-sin(s), cos(s), 0]), s, 0, 2 * pi))
