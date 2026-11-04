# Exercise exr-forms-r3: forms on R^3 with SageManifolds and the dictionary with vector calculus: d of a function is the
# gradient, d of a one-form is the curl, d of a two-form is the divergence, and d^2 = 0 encodes curl grad = 0 and
# div curl = 0.
M = Manifold(3, "R^3"); X = M.chart("x y z"); x, y, z = X[:]
f = M.scalar_field(x**2 * y + z, name="f")
df = f.differential(); print("df =", df.display())
om = M.one_form(y * z, x * z, x * y, name="omega")             # components of the vector field (yz, xz, xy)
dom = om.exterior_derivative(); print("d omega =", dom.display(), " -> this vector field is a gradient (curl 0):", dom == 0)
al = M.one_form(-y, x, 0, name="alpha")
dal = al.exterior_derivative(); print("d alpha =", dal.display(), " -> curl of (-y, x, 0) is (0, 0, 2)")
print("d(df) = 0 (curl grad = 0):", df.exterior_derivative() == 0, "; d(d alpha) = 0 (div curl = 0):", dal.exterior_derivative() == 0)
be = M.diff_form(2, name="beta"); be[0, 1] = z; be[0, 2] = -y; be[1, 2] = x       # beta = z dx^dy - y dx^dz + x dy^dz
print("d beta =", be.exterior_derivative().display(), " -> the divergence of (x, y, z) is 3")
