# Exercise exr-compatible-polar-check: nabla g = 0 for the polar connection of the Euclidean metric, while the zero
# connection of the polar chart has nabla g != 0 (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('r ph', lambda c: [[1, 0], [0, c[0]**2]], ranges='r:(0,+oo) ph:(0,2*pi)')
r, ph = X[:]
nab = g.connection()
print("Levi-Civita coefficients:", nab.display())
print("nabla g = 0 for the Levi-Civita connection:", nab(g) == 0)
zero = M.affine_connection('nabla0')                         # Gamma = 0 in the polar chart
zero[0, 0, 0] = 0
dg = zero(g)                                                 # dg[i, j, k] = nabla_k g_ij
print("nabla g for the zero connection of the polar chart:", dg.display())
print("the only non-zero component is nabla_r g_phph = 2r:", bool(dg[1, 1, 0].expr() == 2*r) and all(bool(dg[i, j, k].expr() == 0) for i in range(2) for j in range(2) for k in range(2) if (i, j, k) != (1, 1, 0)))
print("the zero connection is torsion-free:", zero.torsion() == 0, "and flat:", zero.riemann() == 0)
