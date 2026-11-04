# Exercise exr-gr-int-de-sitter-three-ways: the static patch of de Sitter space, ds^2 = (1 - H^2 r^2) dt^2
# - dr^2/(1 - H^2 r^2) - r^2 dOmega^2 (signature (+, -, -, -)), has R_mu nu = -3 H^2 g_mu nu: the same vacuum solution with
# Lambda = 3 H^2 as the flat slicing, here with a horizon at r = 1/H. In the flat slicing the Friedmann equation with
# rho_Lambda = Lambda/(8 pi G) gives H^2 = Lambda/3 again.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
H = var('H'); assume(H > 0)
M, X, g = chart_metric('t r th ph', lambda c: [[1 - H^2*c[1]^2, 0, 0, 0], [0, -1/(1 - H^2*c[1]^2), 0, 0], [0, 0, -c[1]^2, 0], [0, 0, 0, -c[1]^2*sin(c[2])^2]], structure='Lorentzian', ranges='t r:(0,+oo) th:(0,pi) ph:(0,2*pi)')
print("static patch: Ricci = -3 H^2 g:", g.ricci() + 3*H^2*g == 0, "; R =", g.ricci_scalar().expr().simplify_full())
G, Lam = var('G Lam'); rhoL = Lam/(8*pi*G)
print("Friedmann H^2 = 8 pi G rho_Lambda/3 gives H^2 =", 8*pi*G*rhoL/3)
