# Exercise exr-su2-curvature-sage: Ricci tensor and scalar curvature of the SU(2) metric in Euler angles with
# SageManifolds: Ric = 2 g and R = 6, the values of the unit three-sphere.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph ps', lambda c: [[QQ(1)/4, 0, 0], [0, QQ(1)/4, cos(c[0])/4], [0, cos(c[0])/4, QQ(1)/4]], ranges='th:(0,pi) ph:(0,2*pi) ps:(0,4*pi)')
th, ph, ps = X[:]
Ric = g.ricci(); Rs = g.ricci_scalar()
print("Ricci tensor:", Ric.display())
print("R_thth = R_phph = R_psps = 1/2, R_phps = cos(th)/2, R_thph = R_thps = 0:", bool(Ric[0, 0].expr() == 1/2 and Ric[1, 1].expr() == 1/2 and Ric[2, 2].expr() == 1/2 and Ric[1, 2].expr() == cos(th)/2 and Ric[0, 1].expr() == 0 and Ric[0, 2].expr() == 0))
print("Ric = 2 g:", Ric == 2*g, "; scalar curvature R =", Rs.expr(), "= n (n - 1) for n = 3:", bool(Rs.expr() == 6))
Rd = g.riemann().down(g)
secs = [(Rd[a, b, a, b].expr()/(g[a, a].expr()*g[b, b].expr() - g[a, b].expr()**2)).simplify_full() for (a, b) in [(0, 1), (0, 2), (1, 2)]]
print("sectional curvatures of the three coordinate planes:", secs, "-> all equal to 1:", all(bool(s == 1) for s in secs))
