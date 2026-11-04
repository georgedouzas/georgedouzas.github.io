# Exercise exr-gauss-gnomonic: the central projection of the unit sphere onto its tangent plane at the north pole gives
# the metric (du^2 + dv^2)/(1 + r^2) - (u du + v dv)^2/(1 + r^2)^2, of Gauss curvature 1, whose geodesics are the
# straight lines of the chart plane; the line v = 0 is traversed as u = tan s.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
th, ph = var('th ph', domain='real'); assume(th > 0, th < pi/2)
u0, v0 = tan(th)*cos(ph), tan(th)*sin(ph)                 # (X/Z, Y/Z) on the upper hemisphere
def gnomonic(c):
    q = 1 + c[0]**2 + c[1]**2
    return [[1/q - c[0]**2/q**2, -c[0]*c[1]/q**2], [-c[0]*c[1]/q**2, 1/q - c[1]**2/q**2]]
M, X, g = chart_metric('u v', gnomonic)
J = matrix([[diff(u0, th), diff(u0, ph)], [diff(v0, th), diff(v0, ph)]])
G = matrix(SR, 2, 2, lambda i, j: g[i, j].expr().subs({X[0]: u0, X[1]: v0}))
pb = (J.transpose()*G*J).simplify_full()
print("pulled back to (theta, phi):", pb.list())
Rd = g.riemann().down(g)
print("Gauss curvature K =", (Rd[0, 1, 0, 1].expr()/g.determinant().expr()).simplify_full())
Gam = g.christoffel_symbols(); s = var('s')
uu = tan(s)
print("u = tan s, v = 0 solves the geodesic equation:", bool((diff(uu, s, 2) + Gam[0, 0, 0].expr().subs({X[0]: uu, X[1]: 0})*diff(uu, s)**2).simplify_full() == 0),
      "; Gamma^v_uu on v = 0:", Gam[1, 0, 0].expr().subs({X[1]: 0}))
b, lam = var('b'), function('lam')(s)
ode = diff(lam, s, 2) + Gam[0, 0, 0].expr().subs({X[0]: lam, X[1]: b})*diff(lam, s)**2
print("on a line v = b: Gamma^v_uu =", Gam[1, 0, 0].expr(), "everywhere, so the v-equation holds; the u-equation is the ODE", ode.simplify_full(), "= 0")
