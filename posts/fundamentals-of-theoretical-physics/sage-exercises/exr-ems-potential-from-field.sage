# Exercise exr-ems-potential-from-field: potential of E = (yz, xz, xy) by line integrals along two paths.
x, y, z, t = var('x y z t')
E = vector([y * z, x * z, x * y])
curl = vector([diff(E[2], y) - diff(E[1], z), diff(E[0], z) - diff(E[2], x), diff(E[1], x) - diff(E[0], y)])
print("curl E =", curl)
# path 1: straight line t (x, y, z)
p1 = -integrate(E.subs(x=t * x, y=t * y, z=t * z).dot_product(vector([x, y, z])), t, 0, 1)
# path 2: along x, then y, then z
p2 = -(integrate(E[0].subs(x=t, y=0, z=0), t, 0, x) + integrate(E[1].subs(x=x, y=t, z=0), t, 0, y) + integrate(E[2].subs(x=x, y=y, z=t), t, 0, z))
print("path 1:", p1.simplify_full(), "; path 2:", p2.simplify_full())
phi = p1.simplify_full()
print("Laplacian of phi:", sum(diff(phi, v, 2) for v in [x, y, z]), "-> rho = 0")
