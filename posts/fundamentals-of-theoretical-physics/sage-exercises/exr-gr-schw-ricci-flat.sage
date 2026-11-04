# Exercise exr-gr-schw-ricci-flat: the Schwarzschild metric (signature (+, -, -, -), G = 1) has non-zero Riemann tensor
# but vanishing Ricci tensor; the radial tidal component R^r_{t r t} and the scalar curvature are printed.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
m = var('m'); assume(m > 0)
M, X, g = chart_metric('t r th ph', lambda c: [[1 - 2*m/c[1], 0, 0, 0], [0, -1/(1 - 2*m/c[1]), 0, 0], [0, 0, -c[1]^2, 0], [0, 0, 0, -c[1]^2*sin(c[2])^2]], structure='Lorentzian', ranges='t r:(0,+oo) th:(0,pi) ph:(0,2*pi)')
Rm, Ric, Rs = curvature(g)
print("R^r_{t r t} =", Rm[1, 0, 1, 0].expr().simplify_full())
print("Ricci tensor vanishes:", Ric == 0, "; scalar curvature:", Rs.expr())
