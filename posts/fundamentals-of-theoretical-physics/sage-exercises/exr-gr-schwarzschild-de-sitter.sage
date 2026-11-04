# Exercise exr-gr-schwarzschild-de-sitter: the metric with f = 1 - 2m/r - Lambda r^2/3, ds^2 = f dt^2 - dr^2/f - r^2 dOmega^2
# (signature (+, -, -, -)), solves the vacuum equations with cosmological constant: R_mu nu = -Lambda g_mu nu.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
m, Lam = var('m Lam')
M, X, g = chart_metric('t r th ph', lambda c: [[1 - 2*m/c[1] - Lam*c[1]^2/3, 0, 0, 0], [0, -1/(1 - 2*m/c[1] - Lam*c[1]^2/3), 0, 0], [0, 0, -c[1]^2, 0], [0, 0, 0, -c[1]^2*sin(c[2])^2]], structure='Lorentzian', ranges='t r:(0,+oo) th:(0,pi) ph:(0,2*pi)')
Ric = g.ricci()
print("R_mu nu + Lambda g_mu nu = 0:", Ric + Lam*g == 0, "; R =", g.ricci_scalar().expr().simplify_full())
