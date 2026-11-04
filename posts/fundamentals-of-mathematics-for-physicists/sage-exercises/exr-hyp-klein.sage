# Exercise exr-hyp-klein: central projection of the hyperboloid t^2 - x^2 - y^2 = 1 from the origin onto the plane
# t = 1 gives the Beltrami-Klein disc, with metric (du^2 + dv^2)/(1 - r^2) + (u du + v dv)^2/(1 - r^2)^2, Gauss
# curvature -1, and straight chords as geodesics.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
psi, ph = var('psi ph', domain='real'); assume(psi > 0)
u0, v0 = tanh(psi)*cos(ph), tanh(psi)*sin(ph)          # (x/t, y/t) on the upper sheet
def klein(c):
    s = 1 - c[0]**2 - c[1]**2
    return [[1/s + c[0]**2/s**2, c[0]*c[1]/s**2], [c[0]*c[1]/s**2, 1/s + c[1]**2/s**2]]
M, X, g = chart_metric('u v', klein)
J = matrix([[diff(u0, psi), diff(u0, ph)], [diff(v0, psi), diff(v0, ph)]])
G = matrix(SR, 2, 2, lambda i, j: g[i, j].expr().subs({X[0]: u0, X[1]: v0}))
pb = (J.transpose()*G*J).simplify_full()
print("pulled back to (psi, phi):", pb.list(), "= d psi^2 + sinh^2 psi d phi^2:", bool(pb[0, 0] == 1) and bool((pb[1, 1] - sinh(psi)**2).simplify_full() == 0) and bool(pb[0, 1] == 0))
Rd = g.riemann().down(g)
K = (Rd[0, 1, 0, 1].expr()/g.determinant().expr()).simplify_full()
print("Gauss curvature K =", K)
s = var('s')
Gam = g.christoffel_symbols()
uu = tanh(s)                                    # the diameter v = 0, at unit speed
acc = diff(uu, s, 2) + Gam[0, 0, 0].expr().subs({X[0]: uu, X[1]: 0})*diff(uu, s)**2
print("u = tanh s, v = 0 solves the geodesic equation:", bool(acc.simplify_full() == 0))
# a horizontal chord v = b, u = lam(s): the v-equation is Gamma^v_uu lam'^2 = 0, the u-equation an ODE for lam
b, lam = var('b'), function('lam')(s)
print("Gamma^v_uu =", Gam[1, 0, 0].expr(), "; Gamma^v_uv on v = b:", Gam[1, 0, 1].expr().subs({X[1]: b}).simplify_full(), "(multiplies v' = 0)")
ode = diff(lam, s, 2) + Gam[0, 0, 0].expr().subs({X[0]: lam, X[1]: b})*diff(lam, s)**2
print("on v = b the u-equation is the ODE", ode.simplify_full(), "= 0, solvable for lam: every chord is a geodesic")
