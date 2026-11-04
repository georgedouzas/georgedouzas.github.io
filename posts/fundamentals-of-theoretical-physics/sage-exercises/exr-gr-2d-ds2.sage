# Exercise exr-gr-2d-ds2: two-dimensional de Sitter space ds^2 = dt^2 - e^{2Ht} dx^2 (signature (+, -)) is curved,
# R = -2 H^2, yet its Einstein tensor vanishes.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
H = var('H'); assume(H > 0)
M, X, g = chart_metric('t x', lambda c: [[1, 0], [0, -exp(2*H*c[0])]], structure='Lorentzian')
Ric, Rs = g.ricci(), g.ricci_scalar()
print("Ricci = -H^2 g:", Ric + H^2*g == 0, "; R =", Rs.expr().simplify_full(), "; G = Ric - R g/2 = 0:", Ric - (Rs/2)*g == 0)
