# Exercise exr-gr-flrw-conformal: in conformal time eta = -e^{-Ht}/H (eta < 0) de Sitter space is
# ds^2 = (d eta^2 - dx^2)/(H eta)^2, conformally flat, and SageManifolds confirms Ricci = -3 H^2 g in these coordinates.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
H = var('H'); assume(H > 0)
M, X, g = chart_metric('eta x y z', lambda c: [[1/(H*c[0])^2, 0, 0, 0], [0, -1/(H*c[0])^2, 0, 0], [0, 0, -1/(H*c[0])^2, 0], [0, 0, 0, -1/(H*c[0])^2]], structure='Lorentzian', ranges='eta:(-oo,0) x y z')
print("Ricci = -3 H^2 g:", g.ricci() + 3*H^2*g == 0)
t = var('t'); eta = -exp(-H*t)/H
print("a(t) = e^{Ht} equals 1/(-H eta):", bool(exp(H*t) == 1/(-H*eta)), "; d eta/dt = 1/a:", bool(diff(eta, t) == exp(-H*t)))
