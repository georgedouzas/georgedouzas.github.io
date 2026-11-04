# Exercise exr-su2-christoffel-sage: the Christoffel symbols of the SU(2) metric in Euler angles with SageManifolds: five
# of the eighteen independent symbols are non-zero, matching the proposition.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph ps', lambda c: [[QQ(1)/4, 0, 0], [0, QQ(1)/4, cos(c[0])/4], [0, cos(c[0])/4, QQ(1)/4]], ranges='th:(0,pi) ph:(0,2*pi) ps:(0,4*pi)')
th, ph, ps = X[:]
print("inverse metric:", g.inverse().display())
G = christoffel(g)
print("non-zero Christoffel symbols Gamma^k_ij (i <= j):"); [print("  ", k, "=", v) for k, v in G.items()]
expected = {('th', 'ph', 'ps'): sin(th)/2, ('ph', 'th', 'ph'): cos(th)/(2*sin(th)), ('ph', 'th', 'ps'): -1/(2*sin(th)),
            ('ps', 'th', 'ph'): -1/(2*sin(th)), ('ps', 'th', 'ps'): cos(th)/(2*sin(th))}
print("exactly five non-zero symbols, with the values of the proposition:", len(G) == 5 and all(bool((G[k] - expected[k]).simplify_full() == 0) for k in expected))
Gam = g.christoffel_symbols()
print("the psi-circles are geodesics, Gamma^k_psps = 0 for all k:", all(bool(Gam[k, 2, 2].expr() == 0) for k in range(3)))
