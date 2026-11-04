# Exercise exr-gauss-bonnet: the Gauss curvature of the unit sphere integrates to 4 pi (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
Rm, Ric, Rs = curvature(g); th, ph = X[:]
K = Rs.expr()/2
print("Gauss curvature K = R/2 =", K)
print("integral of K dA over the sphere:", integrate(integrate(K*g.sqrt_abs_det().expr(), ph, 0, 2*pi), th, 0, pi))
