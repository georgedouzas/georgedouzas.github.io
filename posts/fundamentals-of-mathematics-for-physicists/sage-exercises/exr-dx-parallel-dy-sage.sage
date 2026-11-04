# Exercise exr-dx-parallel-dy-sage: the constant field d/dy, with polar components (sin phi, cos phi / r), is covariantly
# constant, while the coordinate field d/dr is not (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('r ph', lambda c: [[1, 0], [0, c[0]**2]], ranges='r:(0,+oo) ph:(0,2*pi)')
r, ph = X[:]
nab = g.connection()
Y = M.vector_field(sin(ph), cos(ph)/r, name='Y')             # d/dy in polar components
DY = nab(Y)
print("nabla (d/dy) =", DY.display())
print("d/dy is covariantly constant:", DY == 0)
E_r = M.vector_field(1, 0, name='e_r')                       # d/dr
DE = nab(E_r)
print("nabla (d/dr) =", DE.display())
print("nabla_ph (d/dr) = (1/r) d/dph, nabla_r (d/dr) = 0:", bool(DE[1, 1].expr() == 1/r and DE[0, 1].expr() == 0 and DE[0, 0].expr() == 0 and DE[1, 0].expr() == 0))
a, b = var('a b', domain='real')
Z = a*M.vector_field(cos(ph), -sin(ph)/r) + b*Y             # a d/dx + b d/dy
print("every constant field a d/dx + b d/dy is covariantly constant:", nab(Z) == 0)
