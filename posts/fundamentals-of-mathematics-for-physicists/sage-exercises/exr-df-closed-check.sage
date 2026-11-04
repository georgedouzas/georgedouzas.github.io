# Exercise exr-df-closed-check: which one-forms are closed: (2xy + z) dx + x^2 dy + x dz on R^3 (closed, and exact with
# potential x^2 y + x z), y dx - x dy (not closed) and (x dy - y dx)/(x^2 + y^2) (closed on the punctured plane).
M = Manifold(3, "R^3"); cart = M.chart("x y z"); x, y, z = cart[:]
w1 = M.one_form(2*x*y + z, x**2, x, name="w1")
w2 = M.one_form(y, -x, 0, name="w2")
w3 = M.one_form(-y/(x**2 + y**2), x/(x**2 + y**2), 0, name="w3")
for w in [w1, w2, w3]:
    print("d%s =" % w._name, w.exterior_derivative().display())
f = M.scalar_field(x**2*y + x*z, name="f")
print("w1 = d(x^2 y + x z):", f.differential() == w1)
