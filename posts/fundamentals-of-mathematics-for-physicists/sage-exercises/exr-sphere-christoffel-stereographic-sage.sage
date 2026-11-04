# Exercise exr-sphere-christoffel-stereographic-sage: Christoffel symbols of the round sphere in the stereographic chart,
# g = 4 (du^2 + dv^2) / (1 + u^2 + v^2)^2, regular at the pole u = v = 0, and the conformal formula (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('u v', lambda c: [[4/(1 + c[0]**2 + c[1]**2)**2, 0], [0, 4/(1 + c[0]**2 + c[1]**2)**2]])
u, v = X[:]
G = christoffel(g)
print("non-zero Christoffel symbols Gamma^k_ij (i <= j):"); [print("  ", k, "=", v_) for k, v_ in G.items()]
sigma = log(2/(1 + u**2 + v**2))                               # g = e^{2 sigma} delta
Gam = g.christoffel_symbols(); coords = [u, v]
conformal = lambda k, i, j: (1 if k == i else 0)*diff(sigma, coords[j]) + (1 if k == j else 0)*diff(sigma, coords[i]) - (1 if i == j else 0)*diff(sigma, coords[k])
print("conformal formula Gamma^k_ij = delta^k_i d_j sigma + delta^k_j d_i sigma - delta_ij d^k sigma:", all(bool((Gam[k, i, j].expr() - conformal(k, i, j)).simplify_full() == 0) for k in range(2) for i in range(2) for j in range(2)))
print("all symbols vanish at the pole u = v = 0:", all(bool(Gam[k, i, j].expr().subs(u=0, v=0) == 0) for k in range(2) for i in range(2) for j in range(2)))
print("Gauss curvature K = R/2 = 1 in this chart:", bool((g.ricci_scalar().expr()/2).simplify_full() == 1))
