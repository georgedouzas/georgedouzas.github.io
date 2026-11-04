# Exercise exr-parallel-field-meridian-sage: parallel transport along a meridian of the unit sphere, solved with the
# Christoffel symbols from SageManifolds: V^theta is constant, V^phi = c / sin(t), and the norm is preserved.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
Gam = g.christoffel_symbols(); th, ph = X[:]
t = var('t', domain='real'); assume(0 < t, t < pi)
a, b = var('a b', domain='real')
# the meridian theta = t, phi = phi0 has velocity (1, 0); transport: V'^k + Gamma^k_{theta j} V^j = 0
print("Gamma^theta_{theta j} =", [Gam[0, 0, j].expr() for j in range(2)], ";  Gamma^phi_{theta j} =", [Gam[1, 0, j].expr().subs(th=t) for j in range(2)])
Vth = function('Vth')(t); Vph = function('Vph')(t)
Vth_t = desolve(diff(Vth, t) + Gam[0, 0, 0].expr().subs(th=t)*Vth == 0, Vth, ics=[pi/2, a])        # Gamma^theta_{theta phi} = 0 decouples it
Vph_t = desolve(diff(Vph, t) + Gam[1, 0, 1].expr().subs(th=t)*Vph == 0, Vph, ics=[pi/2, b])        # Gamma^phi_{theta theta} = 0 decouples it
Vph_t = Vph_t.simplify_full()
print("V^theta(t) =", Vth_t, ";  V^phi(t) =", Vph_t, "  (starting from (a, b) on the equator)")
print("V^phi sin(t) is constant:", bool((Vph_t*sin(t)).simplify_full() == b))
print("norm g(V, V) = a^2 + b^2 preserved:", bool((Vth_t**2 + sin(t)**2*Vph_t**2).simplify_full() == a**2 + b**2))
