# Exercise exr-einstein-2d: the Einstein tensor of ANY two-dimensional metric vanishes identically (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M = Manifold(2, 'M', structure='Riemannian'); X = M.chart('u v'); u, v = X[:]
A = function('A')(u, v); B = function('B')(u, v); C = function('C')(u, v)      # a generic symmetric 2x2 metric
g = M.metric('g'); g[0, 0] = A; g[0, 1] = B; g[1, 1] = C
Ric, Rs = g.ricci(), g.ricci_scalar()
G = Ric - (Rs/2)*g
print("Einstein tensor of a generic 2D metric:", G.display())
print("[G_ij = 0 identically]:", G == 0)
