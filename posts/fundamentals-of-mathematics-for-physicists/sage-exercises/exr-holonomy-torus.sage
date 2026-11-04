# Exercise exr-holonomy-torus: holonomy of a flat U(1) connection on the circle bundle S^1 x S^1 over the circle:
# the horizontal lift of the base loop for A = c dx shifts the fibre angle by -2 pi c, the integral of -A around the
# loop, for a connection whose curvature vanishes (SageManifolds).
T2 = Manifold(2, "T^2"); X = T2.chart("x:(0,2*pi) y:(0,2*pi)"); x, y = X[:]
c = var("c", domain="real")
A = T2.one_form(c, 0, name="A")                              # A = c dx
print("curvature F = dA =", A.exterior_derivative().display(), " -> flat")
t = var("t", domain="real")
yfun = function("yf")(t)
sol = desolve(diff(yfun, t) == -c, yfun, ics=[0, 0], ivar=t)   # horizontal lift: dy/dt = -c dx/dt with x = t
print("horizontal lift of x = t from y(0) = 0: y(t) =", sol, "; y(2 pi) =", sol.subs(t=2*pi), "= -integral of A around the loop:", bool(sol.subs(t=2*pi) == -integrate(c, t, 0, 2*pi)))
print("for c = 1/4 the holonomy is a rotation of the fibre by -pi/2; it is trivial exactly when c is an integer")
