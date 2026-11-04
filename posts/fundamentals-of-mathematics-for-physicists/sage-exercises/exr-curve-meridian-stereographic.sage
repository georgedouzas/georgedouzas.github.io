# Exercise exr-curve-meridian-stereographic: the great circle t -> (cos t, 0, sin t) through the poles, seen in the north
# stereographic chart as the curve (u, v) = (cos t/(1 - sin t), 0); its velocity du/dt = 1/(1 - sin t), and the chain
# rule d/dt Z(gamma(t)) = (du/dt) dZ/du + (dv/dt) dZ/dv checked against dZ/dt = cos t.
t = var("t", domain="real")
S2 = Manifold(2, "S^2"); U = S2.open_subset("U"); ste = U.chart("u v"); u, v = ste[:]
X, Y, Z = cos(t), 0, sin(t)
uc, vc = X/(1 - Z), Y/(1 - Z)                                # north stereographic coordinates of the curve
gamma = U.curve({ste: [uc, vc]}, (t, -pi/2, pi/2), name="gamma")   # the half of the great circle below the equator plane... up to the pole
print("gamma in the chart:", gamma.coord_expr(ste))
E = [2*u/(1 + u**2 + v**2), 2*v/(1 + u**2 + v**2), (u**2 + v**2 - 1)/(1 + u**2 + v**2)]
print("the chart point lies back on (cos t, 0, sin t):", all(bool((e.subs(u=uc, v=vc) - p).simplify_full() == 0) for e, p in zip(E, (X, Y, Z))))
udot, vdot = diff(uc, t).simplify_full(), diff(vc, t)
print("velocity (du/dt, dv/dt) =", (udot, vdot), " and du/dt = 1/(1 - sin t):", bool((udot - 1/(1 - sin(t))).simplify_full() == 0))
Zf = U.scalar_field({ste: E[2]}, name="Z")
chain = (udot*diff(Zf.expr(ste), u) + vdot*diff(Zf.expr(ste), v)).subs(u=uc, v=vc)
print("chain rule (du/dt) dZ/du + (dv/dt) dZ/dv = cos t = dZ/dt:", bool((chain - diff(Z, t)).simplify_full() == 0))
print("du/dt blows up as t -> pi/2, where the curve leaves the chart through the north pole:", limit(udot, t=pi/2, dir="-"))
print("at t = -pi/2 the curve passes the south pole, u =", uc.subs(t=-pi/2), " with speed du/dt =", udot.subs(t=-pi/2))
