# Exercise exr-fbun-lift-constant-field: on the trivial circle bundle over the plane with potential
# A = (x dy - y dx)/2 (curvature F = dx ^ dy), the horizontal lift of the circle of radius R traversed once
# counterclockwise shifts the fibre angle by -pi R^2: minus the enclosed area, the flux.
M = Manifold(2, "R2"); X = M.chart("x y"); x, y = X[:]
A = M.one_form(-y / 2, x / 2, name="A")
print("F = dA =", A.exterior_derivative().display())
t, R = var("t R", domain="real"); assume(R > 0)
xc, yc = R * cos(t), R * sin(t)
Adot = (A[0].expr().subs(x=xc, y=yc) * diff(xc, t) + A[1].expr().subs(x=xc, y=yc) * diff(yc, t)).simplify_full()
chi = function("chi")(t)
sol = desolve(diff(chi, t) == -Adot, chi, ics=[0, 0], ivar=t)
print("horizontal lift: chi(t) =", sol)
print("holonomy chi(2 pi) =", sol.subs(t=2 * pi), "; equals -pi R^2:", bool(sol.subs(t=2 * pi) == -pi * R ** 2))
flux = integrate(integrate(1, y, -sqrt(R ** 2 - x ** 2), sqrt(R ** 2 - x ** 2)), x, -R, R)
print("flux of F through the disc:", flux)
