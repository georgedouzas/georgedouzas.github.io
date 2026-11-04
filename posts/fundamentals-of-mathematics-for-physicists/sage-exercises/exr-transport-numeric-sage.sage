# Exercise exr-transport-numeric-sage: numerical parallel transport of d/dtheta around the parallel theta = pi/3 of the
# unit sphere (Runge-Kutta on the transport equation); the vector returns rotated by 2 pi cos(pi/3) = pi, and its norm
# stays 1 along the way.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
Gam = g.christoffel_symbols(); th, ph = X[:]
th0 = pi/3
t, vth, vph = var('t vth vph')
# curve theta = th0, phi = t; velocity (0, 1); dV^k/dt = -Gamma^k_{ph j} V^j
rhs = [SR(-(Gam[0, 1, 0].expr()*vth + Gam[0, 1, 1].expr()*vph).subs(th=th0)), SR(-(Gam[1, 1, 0].expr()*vth + Gam[1, 1, 1].expr()*vph).subs(th=th0))]
print("dV^theta/dt =", rhs[0], ";  dV^phi/dt =", rhs[1])
sol = desolve_system_rk4(rhs, [vth, vph], ics=[0, 1, 0], ivar=t, end_points=float(2*pi), step=0.01)
T, Vth, Vph = sol[-1]
print("after one turn (t = %.4f): V^theta = %.6f, V^phi = %.6f" % (T, Vth, Vph))
angle = float(atan2(float(sin(th0))*Vph, Vth))                         # angle in the orthonormal frame (e_theta, e_phi)
print("rotation angle |%.6f| = pi = 2 pi cos(theta0):" % angle, abs(abs(angle) - float(pi)) < 1e-4)
norms = [float(sqrt(a**2 + float(sin(th0))**2*b**2)) for (_, a, b) in sol[::50]]
print("norm along the way:", ["%.5f" % n for n in norms][:6], "... all within 1e-5 of 1:", all(abs(n - 1) < 1e-5 for n in norms))
print("holonomy angle 2 pi (1 - cos theta0) = pi equals the area of the cap:", bool(2*pi*(1 - cos(th0)) == pi))
