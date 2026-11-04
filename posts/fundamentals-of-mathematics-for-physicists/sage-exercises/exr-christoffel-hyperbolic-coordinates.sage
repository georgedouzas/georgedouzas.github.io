# Exercise exr-christoffel-hyperbolic-coordinates: the Euclidean plane in the coordinates x = s e^t, y = s e^(-t)
# has the metric 2 cosh(2t) ds^2 + 4 s sinh(2t) ds dt + 2 s^2 cosh(2t) dt^2 with determinant 4 s^2, and only two
# non-zero Christoffel symbols, Gamma^s_tt = s and Gamma^t_st = 1/s; the coordinates y = v^2 give Gamma^v_vv = 1/v.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
s, t = var('s t', domain='real')
x, y = s*exp(t), s*exp(-t)
J = matrix([[diff(x, s), diff(x, t)], [diff(y, s), diff(y, t)]])
G = J.transpose()*J
C2, S2 = (exp(2*t) + exp(-2*t))/2, (exp(2*t) - exp(-2*t))/2          # cosh 2t, sinh 2t
target = matrix([[2*C2, 2*s*S2], [2*s*S2, 2*s**2*C2]])
print("pulled-back metric:", [e.simplify_full() for e in G.list()])
print("equals 2cosh(2t) ds^2 + 4 s sinh(2t) ds dt + 2 s^2 cosh(2t) dt^2:", (G - target).simplify_full() == 0)
M, X, g = chart_metric('s t', lambda c: [[2*cosh(2*c[1]), 2*c[0]*sinh(2*c[1])], [2*c[0]*sinh(2*c[1]), 2*c[0]**2*cosh(2*c[1])]])
print("det g =", g.determinant().expr().simplify_full())
print("non-zero Christoffel symbols:", christoffel(g))
print("flat:", g.riemann() == 0)
M2, X2, g2 = chart_metric('u v', lambda c: [[1, 0], [0, 4*c[1]**2]])
print("coordinates (u, v) with y = v^2: metric du^2 + 4 v^2 dv^2, symbols", christoffel(g2))
