# Exercise exr-polar-christoffel-spherical-sage: the Christoffel symbols of the Euclidean metric of R^3 in spherical
# coordinates (rho, theta, phi); six independent symbols are non-zero and the Riemann tensor vanishes (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('rho th ph', lambda c: [[1, 0, 0], [0, c[0]**2, 0], [0, 0, c[0]**2*sin(c[1])**2]], ranges='rho:(0,+oo) th:(0,pi) ph:(0,2*pi)')
rho, th, ph = X[:]
G = christoffel(g)
print("non-zero Christoffel symbols Gamma^k_ij (i <= j):"); [print("  ", k, "=", v) for k, v in G.items()]
expected = {('rho', 'th', 'th'): -rho, ('rho', 'ph', 'ph'): -rho*sin(th)**2, ('th', 'rho', 'th'): 1/rho,
            ('th', 'ph', 'ph'): -sin(th)*cos(th), ('ph', 'rho', 'ph'): 1/rho, ('ph', 'th', 'ph'): cos(th)/sin(th)}
print("six non-zero symbols, as expected:", len(G) == 6 and all(bool((G[k] - expected[k]).simplify_full() == 0) for k in expected))
print("the Riemann tensor vanishes (flat space in curvilinear coordinates):", g.riemann() == 0)
