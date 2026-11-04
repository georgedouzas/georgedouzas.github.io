# Exercise exr-gr-rindler: the Rindler metric ds^2 = x^2 dt^2 - dx^2 - dy^2 - dz^2 (x > 0, signature (+, -, -, -)) is
# flat: its Riemann tensor vanishes, although its Christoffel symbols do not; T = x sinh t, X = x cosh t maps it to
# Minkowski space.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('t x:(0,+oo) y z', lambda c: [[c[1]^2, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], structure='Lorentzian', ranges='t x:(0,+oo) y z')
print("non-zero Christoffel symbols:", christoffel(g))
print("Riemann tensor vanishes:", g.riemann() == 0)
t, x = var('t x')
T, Xc = x*sinh(t), x*cosh(t)
dT = [diff(T, t), diff(T, x)]; dX = [diff(Xc, t), diff(Xc, x)]
G2 = matrix(SR, 2, 2, lambda i, j: dT[i]*dT[j] - dX[i]*dX[j]).simplify_full()
print("dT^2 - dX^2 in (t, x):", G2, "; equals diag(x^2, -1):", G2 == matrix([[x^2, 0], [0, -1]]))
