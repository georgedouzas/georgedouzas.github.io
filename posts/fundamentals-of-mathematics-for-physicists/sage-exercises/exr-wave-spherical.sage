# Exercise exr-wave-spherical: u = f(r - t)/r solves u_tt - Laplacian u = 0 in three dimensions for generic f, and
# f(r - t) alone does not.
t, x, y, z = var("t x y z")
f = function("f")
r = sqrt(x ** 2 + y ** 2 + z ** 2)
box = lambda u: diff(u, t, 2) - diff(u, x, 2) - diff(u, y, 2) - diff(u, z, 2)
u1 = f(r - t) / r
print("Box[f(r - t)/r] =", box(u1).simplify_full())
u2 = f(r - t)
print("Box[f(r - t)] =", box(u2).simplify_full(), "  (non-zero: the 1/r factor is needed)")
print("spherical wave solves the wave equation:", bool(box(u1).simplify_full() == 0), "| without 1/r:", bool(box(u2).simplify_full() == 0))
