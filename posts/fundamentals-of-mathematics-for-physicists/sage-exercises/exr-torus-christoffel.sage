# Exercise exr-torus-christoffel: Christoffel symbols of the torus metric ds^2 = r^2 da^2 + (R + r cos a)^2 db^2.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
R, r = var('R r', domain='real'); assume(R > r, r > 0)
M, X, g = chart_metric('a b', lambda c: [[r**2, 0], [0, (R + r*cos(c[0]))**2]], ranges='a:(0,2*pi) b:(0,2*pi)')
print("non-zero Christoffel symbols Gamma^k_ij (i <= j):", christoffel(g))
