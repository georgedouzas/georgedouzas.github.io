# Exercise exr-ems-legendre-inside: potential inside the sphere r = R with boundary values V0 cos(theta)^2.
x, y, z, V0, R = var('x y z V0 R'); assume(R > 0)
phi = V0 / 3 + V0 / (3 * R^2) * (2 * z^2 - x^2 - y^2)
print("Laplacian:", sum(diff(phi, v, 2) for v in [x, y, z]))
th, ph = var('theta ph')
bnd = phi.subs(x=R * sin(th) * cos(ph), y=R * sin(th) * sin(ph), z=R * cos(th)).simplify_full()
print("boundary values minus V0 cos^2:", (bnd - V0 * cos(th)^2).simplify_full())
print("value at the centre:", phi.subs(x=0, y=0, z=0), "= average of the boundary values:", integrate(V0 * cos(th)^2 * sin(th), th, 0, pi) / 2)
