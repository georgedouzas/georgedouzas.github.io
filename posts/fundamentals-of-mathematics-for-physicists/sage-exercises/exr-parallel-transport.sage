# Exercise exr-parallel-transport: parallel transport around the parallel theta = theta0 of the unit sphere, with the
# Christoffel symbols from SageManifolds and the transport equations solved symbolically: the vector comes back
# rotated by the holonomy angle 2 pi (1 - cos theta0), which equals the enclosed area (Gauss-Bonnet).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
Gam = g.christoffel_symbols(); th, ph = X[:]
th0 = var("th0", domain="real"); assume(0 < th0, th0 < pi/2); assume(cos(th0) > 0)
t = var("t", domain="real")
Vth = function("Vth")(t); Vph = function("Vph")(t)
# curve theta = th0, phi = t: velocity (0, 1); transport: V'^k + Gamma^k_{ij} xdot^i V^j = 0
eqs = [diff(Vth, t) + Gam[0, 1, 1].expr().subs(th=th0) * Vph == 0,
       diff(Vph, t) + Gam[1, 1, 0].expr().subs(th=th0) * Vth == 0]
sol = desolve_system(eqs, [Vth, Vph], ics=[0, 1, 0], ivar=t)          # start with V = d/dtheta
Vth_t = sol[0].rhs().simplify_full(); Vph_t = sol[1].rhs().simplify_full()
print("V^theta(t) =", Vth_t, ";  V^phi(t) =", Vph_t)
print("norm is preserved, g(V, V) = 1:", bool((Vth_t**2 + sin(th0)**2 * Vph_t**2).simplify_full() == 1))
Vth_end = Vth_t.subs(t=2*pi); print("after one turn V^theta =", Vth_end.simplify_full(), ": the vector is rotated by the angle 2*pi*cos(theta0)")
print("holonomy angle 2 pi - 2 pi cos(theta0) equals the area of the cap:", bool(integrate(integrate(sin(th), ph, 0, 2*pi), th, 0, th0).simplify_full() == 2*pi*(1 - cos(th0))))
