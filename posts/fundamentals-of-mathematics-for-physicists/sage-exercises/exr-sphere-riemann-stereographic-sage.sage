# Exercise exr-sphere-riemann-stereographic-sage: the Riemann tensor of the round sphere in the stereographic chart is
# regular at the pole and gives the same Gauss curvature K = R_uvuv / det g = 1 everywhere (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('u v', lambda c: [[4/(1 + c[0]**2 + c[1]**2)**2, 0], [0, 4/(1 + c[0]**2 + c[1]**2)**2]])
u, v = X[:]
Rm = g.riemann(); Rd = Rm.down(g)
print("R^u_vuv =", Rm[0, 1, 0, 1].expr().factor(), "   R_uvuv =", Rd[0, 1, 0, 1].expr().factor())
print("R_uvuv = det g = 16/(1 + u^2 + v^2)^4:", bool((Rd[0, 1, 0, 1].expr() - 16/(1 + u**2 + v**2)**4).simplify_full() == 0))
K = (Rd[0, 1, 0, 1].expr()/g.determinant().expr()).simplify_full()
print("Gauss curvature K = R_uvuv / det g =", K, "; at the pole u = v = 0:", Rd[0, 1, 0, 1].expr().subs(u=0, v=0), "/", g.determinant().expr().subs(u=0, v=0))
print("K = 1 everywhere, pole included:", bool(K == 1))
