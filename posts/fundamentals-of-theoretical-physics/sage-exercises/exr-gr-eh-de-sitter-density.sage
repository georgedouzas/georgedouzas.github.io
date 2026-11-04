# Exercise exr-gr-eh-de-sitter-density: on four-dimensional de Sitter space ds^2 = dt^2 - e^{2Ht} dx^2, signature
# (+, -, -, -), the Lagrangian density sqrt(-g)(R + 2 Lambda) of the action with cosmological constant Lambda = 3 H^2
# equals -6 H^2 e^{3Ht}.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
H = var('H'); assume(H > 0)
M, X, g = chart_metric('t x y z', lambda c: [[1, 0, 0, 0], [0, -exp(2*H*c[0]), 0, 0], [0, 0, -exp(2*H*c[0]), 0], [0, 0, 0, -exp(2*H*c[0])]], structure='Lorentzian')
t = X[0]
sqrtg = sqrt(-g.determinant().expr()).simplify_full()
R = g.ricci_scalar().expr().simplify_full()
dens = (sqrtg*(R + 2*3*H^2)).simplify_full()
print("sqrt(-g) =", sqrtg, "; R =", R, "; sqrt(-g)(R + 2 Lambda) =", dens, "; equals -6 H^2 e^{3Ht}:", bool(dens == -6*H^2*exp(3*H*t)))
