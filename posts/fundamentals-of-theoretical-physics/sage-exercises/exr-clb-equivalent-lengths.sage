# Exercise exr-clb-equivalent-lengths (Sinha, Secs. 2.4.2-2.4.3): lengths l = I_O/(M h) of the simple pendulums equivalent
# to compound pendulums, with the moments of inertia computed by integration: a circular wire about a tangent in its
# plane and about a perpendicular axis through a point of the wire, a solid hemisphere about a diameter of its base, and
# a cube of side 2a about an edge and about a diagonal of a face.
r, a, th, x, y, z, rho, ph = var('r a th x y z rho ph'); assume(r > 0, a > 0)
# circular wire of radius r in the xy-plane, centre at the origin, unit mass per length
lam = 1 / (2 * pi * r)
X, Y = r * cos(th), r * sin(th)
I_tangent = integrate(lam * (X - r)^2 * r, th, 0, 2 * pi)            # axis x = r in the plane (distance |X - r|)
I_perp = integrate(lam * ((X - r)^2 + Y^2) * r, th, 0, 2 * pi)       # axis through (r, 0) perpendicular to the plane
print("wire, tangent axis: l =", (I_tangent / r).simplify_full(), "  perpendicular axis: l =", (I_perp / r).simplify_full())
# solid hemisphere z >= 0 of radius a, unit total mass; axis: the x-axis (a diameter of the base)
dens = 1 / (2 * pi * a^3 / 3)
I_h = integrate(integrate(integrate(dens * ((rho * sin(th) * sin(ph))^2 + (rho * cos(th))^2) * rho^2 * sin(th), rho, 0, a), th, 0, pi / 2), ph, 0, 2 * pi)
h_h = integrate(integrate(integrate(dens * rho * cos(th) * rho^2 * sin(th), rho, 0, a), th, 0, pi / 2), ph, 0, 2 * pi)
print("hemisphere: I = %s, h = %s, l = %s" % (I_h, h_h, (I_h / h_h).simplify_full()))
# cube [-a, a]^3 of unit mass; edge: the line y = z = a (parallel to x); face diagonal: x = y in the face z = a
dc = 1 / (2 * a)^3
I_edge = integrate(integrate(integrate(dc * ((y - a)^2 + (z - a)^2), x, -a, a), y, -a, a), z, -a, a)
# distance squared from (x, y, z) to the line through (0, 0, a) with direction (1, 1, 0)/sqrt(2)
d2 = x^2 + y^2 + (z - a)^2 - (x + y)^2 / 2
I_diag = integrate(integrate(integrate(dc * d2, x, -a, a), y, -a, a), z, -a, a)
print("cube, edge: I = %s, h = %s, l = %s" % (I_edge, sqrt(2) * a, (I_edge / (sqrt(2) * a)).simplify_full()))
print("cube, face diagonal: I = %s, h = %s, l = %s" % (I_diag, a, (I_diag / a).simplify_full()))
