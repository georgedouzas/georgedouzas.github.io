# Exercise exr-flow-rotation: the flow of the rotation field of the plane, from the differential equation
# dx/dt = X(x), and its group property; the same for the rotation field of the sphere in the spherical chart.
t, s = var("t s", domain="real")
x = function("x")(t); y = function("y")(t)
x0, y0 = var("x0 y0", domain="real")
sol = desolve_system([diff(x, t) == -y, diff(y, t) == x], [x, y], ics=[0, x0, y0], ivar=t)
Phi = (sol[0].rhs(), sol[1].rhs())
print("Phi_t(x0, y0) =", Phi)
Phi_s_of_Phi_t = tuple(e.subs({x0: Phi[0], y0: Phi[1]}).subs(t=s) if False else e.subs(t=s).subs({x0: Phi[0], y0: Phi[1]}) for e in Phi)
print("group property Phi_s o Phi_t = Phi_{s+t}:", bool(all((a - b.subs(t=s + t)).simplify_full() == 0 for a, b in zip(Phi_s_of_Phi_t, Phi))))
print("Phi_0 is the identity:", [e.subs(t=0) for e in Phi] == [x0, y0])
th = function("th")(t); ph = function("ph")(t)
sol2 = desolve_system([diff(th, t) == 0, diff(ph, t) == 1], [th, ph], ics=[0, var("th0"), var("ph0")], ivar=t)
print("flow of d/dphi on the sphere: (theta, phi) ->", (sol2[0].rhs(), sol2[1].rhs()), " a rotation about the Z-axis by the angle t")
