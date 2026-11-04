# Exercise exr-de-sitter-4d: four-dimensional de Sitter space ds^2 = dt^2 - e^{2Ht} dx^2 in flat slicing (signature
# (+, -, -, -)) solves G - Lambda g = 0 with Lambda = 3 H^2: Ricci = -3 H^2 g, R = -12 H^2.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
H = var('H', domain='real'); assume(H > 0)
M, X, g = chart_metric('t x y z', lambda c: [[1, 0, 0, 0], [0, -exp(2*H*c[0]), 0, 0], [0, 0, -exp(2*H*c[0]), 0], [0, 0, 0, -exp(2*H*c[0])]], structure='Lorentzian')
Ric, Rs = g.ricci(), g.ricci_scalar()
print("Ricci tensor = -3 H^2 g:", (Ric + 3*H**2*g) == 0, "; scalar curvature R =", Rs.expr().simplify_full())
G = Ric - (Rs/2)*g
print("Einstein tensor G - 3 H^2 g = 0 (Lambda = 3 H^2):", (G - 3*H**2*g) == 0)
