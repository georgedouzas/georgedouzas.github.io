# Exercise exr-sphere-curvature-sage: the Ricci tensor and scalar curvature of the unit sphere in the spherical and the
# stereographic charts: Ric = g and R = 2 in both (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
Ric = g.ricci(); Rs = g.ricci_scalar()
print("spherical chart: Ric =", Ric.display(), "; Ric = g:", Ric == g, "; R =", Rs.expr())
N, Y, h = chart_metric('u v', lambda c: [[4/(1 + c[0]**2 + c[1]**2)**2, 0], [0, 4/(1 + c[0]**2 + c[1]**2)**2]])
Rich = h.ricci(); Rsh = h.ricci_scalar()
print("stereographic chart: Ric = g:", Rich == h, "; R =", Rsh.expr().simplify_full())
print("both charts give R = 2:", bool(Rs.expr() == 2) and bool(Rsh.expr().simplify_full() == 2))
