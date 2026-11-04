# Exercise exr-polar-christoffel: Christoffel symbols of the flat plane in polar coordinates (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('r ph', lambda c: [[1, 0], [0, c[0]**2]], ranges='r:(0,+oo) ph:(0,2*pi)')
print("non-zero Christoffel symbols Gamma^k_ij (i <= j):", christoffel(g))
