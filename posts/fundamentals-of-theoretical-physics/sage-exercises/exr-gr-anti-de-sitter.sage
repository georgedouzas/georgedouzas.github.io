# Exercise exr-gr-anti-de-sitter: anti-de Sitter space in Poincare coordinates, ds^2 = (l^2/z^2)(dt^2 - dx^2 - dy^2 - dz^2),
# signature (+, -, -, -): R_mu nu = (3/l^2) g_mu nu, R = 12/l^2, and G - Lambda g = 0 with Lambda = -3/l^2.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
l = var('l'); assume(l > 0)
M, X, g = chart_metric('t x y z', lambda c: [[l^2/c[3]^2, 0, 0, 0], [0, -l^2/c[3]^2, 0, 0], [0, 0, -l^2/c[3]^2, 0], [0, 0, 0, -l^2/c[3]^2]], structure='Lorentzian', ranges='t x y z:(0,+oo)')
Ric, Rs = g.ricci(), g.ricci_scalar()
print("Ricci = (3/l^2) g:", Ric - (3/l^2)*g == 0, "; R =", Rs.expr().simplify_full())
print("G - Lambda g = 0 with Lambda = -3/l^2:", Ric - (Rs/2)*g + (3/l^2)*g == 0)
