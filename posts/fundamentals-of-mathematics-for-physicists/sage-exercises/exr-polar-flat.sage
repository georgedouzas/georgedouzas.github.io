# Exercise exr-polar-flat: the plane in polar coordinates has vanishing Riemann tensor (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('r ph', lambda c: [[1, 0], [0, c[0]**2]], ranges='r:(0,+oo) ph:(0,2*pi)')
Rm, Ric, Rs = curvature(g)
print("Riemann tensor:", Rm.display())
print("all components vanish:", Rm == 0, "; Ricci tensor:", Ric.display(), "; scalar curvature:", Rs.expr())
