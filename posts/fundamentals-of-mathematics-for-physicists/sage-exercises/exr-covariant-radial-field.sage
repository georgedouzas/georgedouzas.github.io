# Exercise exr-covariant-radial-field: the covariant derivative of the radial field r d/dr = x d/dx + y d/dy in polar
# coordinates is the identity tensor, nabla_i X^k = delta^k_i (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('r ph', lambda c: [[1, 0], [0, c[0]**2]], ranges='r:(0,+oo) ph:(0,2*pi)')
r, ph = X[:]
nab = g.connection()
V = M.vector_field(r, 0, name='V')                           # V = r d/dr, the field x d/dx + y d/dy
DV = nab(V)                                                  # components DV[k, i] = nabla_i V^k
print("nabla V =", DV.display())
print("nabla_i V^k = delta^k_i:", all(bool(DV[k, i].expr() == (1 if k == i else 0)) for k in range(2) for i in range(2)))
W = M.vector_field(0, 1, name='W')                           # W = d/dphi, the rotation field
DW = nab(W)
print("nabla W =", DW.display())
print("nabla_r W^ph = 1/r and nabla_ph W^r = -r:", bool(DW[1, 0].expr() == 1/r and DW[0, 1].expr() == -r))
