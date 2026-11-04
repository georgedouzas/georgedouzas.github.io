# Exercise exr-levi-civita-sage-connection: the Levi-Civita connection of the round sphere from SageManifolds' `connection`:
# its coefficients are the Christoffel symbols, it is torsion-free and metric-compatible.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
th, ph = X[:]
nab = g.connection()
print(nab.display())
Gam = g.christoffel_symbols()
print("coefficients of the connection = Christoffel symbols of the metric:", all(bool(nab[k, i, j].expr() == Gam[k, i, j].expr()) for k in range(2) for i in range(2) for j in range(2)))
print("torsion-free:", nab.torsion() == 0, "  metric-compatible (nabla g = 0):", nab(g) == 0)
