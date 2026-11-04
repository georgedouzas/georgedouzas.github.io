# Exercise exr-df-wedge-compute: the wedge product (dx + 2 dy) ^ (3 dx - dz) on R^3, and the commutation of a one-form
# with a two-form, alpha ^ beta = beta ^ alpha.
M = Manifold(3, "R^3"); cart = M.chart("x y z"); x, y, z = cart[:]
dx, dy, dz = cart.coframe()[:]
al = dx + 2*dy; be = 3*dx - dz
print("(dx + 2 dy) ^ (3 dx - dz) =", al.wedge(be).display())
print("equals -6 dx^dy - dx^dz - 2 dy^dz:", al.wedge(be) == -6*dx.wedge(dy) - dx.wedge(dz) - 2*dy.wedge(dz))
a = M.one_form(*[function("a%d" % i)(x, y, z) for i in range(3)])
b = M.diff_form(2); b[0, 1], b[0, 2], b[1, 2] = [function("b%d" % i)(x, y, z) for i in range(3)]
print("one-form ^ two-form = two-form ^ one-form:", a.wedge(b) == b.wedge(a))
