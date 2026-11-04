# Exercise exr-sphere-geodesics-torus-sage: the geodesic equations of the torus metric r^2 da^2 + (R + r cos a)^2 db^2;
# the outer and inner equators and the tube circles satisfy them, a general parallel a = a0 does not (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
R, r = var('R r', domain='real'); assume(R > r, r > 0)
M, X, g = chart_metric('a b', lambda c: [[r**2, 0], [0, (R + r*cos(c[0]))**2]], ranges='a:(0,2*pi) b:(0,2*pi)')
a, b = X[:]; Gam = g.christoffel_symbols()
t = var('t', domain='real')
def residuals(curve):
    """residuals of x''^k + Gamma^k_ij x'^i x'^j = 0 for the curve (a(t), b(t))"""
    sub = {a: curve[0], b: curve[1]}
    return [(diff(curve[k], t, 2) + sum(Gam[k, i, j].expr().subs(sub)*diff(curve[i], t)*diff(curve[j], t) for i in range(2) for j in range(2))).simplify_full() for k in range(2)]
print("geodesic equations: a'' + Gamma^a_bb b'^2 = 0 with Gamma^a_bb =", Gam[0, 1, 1].expr().factor(), ";  b'' + 2 Gamma^b_ab a' b' = 0 with Gamma^b_ab =", Gam[1, 0, 1].expr().factor())
a0 = var('a0', domain='real')
print("outer equator a = 0, b = t:", residuals([SR(0), t]), "; inner equator a = pi, b = t:", residuals([SR(pi), t]))
print("tube circle a = t, b = const:", residuals([t, SR(1)]))
print("parallel a = a0, b = t: residuals", residuals([a0 + 0*t, t]), "-> vanish only where sin(a0) = 0")
