# Exercise exr-gr-einstein-weak-field: for the static weak-field metric g_tt = 1 + 2 eps Phi, g_ij = -(1 - 2 eps Phi)
# delta_ij (signature (+, -, -, -), eps bookkeeps the order), SageManifolds gives G_tt = 2 eps Laplacian(Phi) + O(eps^2),
# so G_tt = 8 pi G rho is Poisson's equation; also R_tt = eps Laplacian(Phi) + O(eps^2).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M = Manifold(4, 'M', structure='Lorentzian'); X = M.chart('t x y z'); t, x, y, z = X[:]
eps = var('eps'); Phi = function('Phi')(x, y, z)
g = M.metric('g'); g[0, 0] = 1 + 2*eps*Phi
for i in (1, 2, 3): g[i, i] = -(1 - 2*eps*Phi)
Ric = g.ricci(); Rs = g.ricci_scalar()
Gtt = (Ric[0, 0] - Rs/2*g[0, 0]).expr()
lap = sum(diff(Phi, c, 2) for c in (x, y, z))
lin = lambda e: (diff(e, eps).subs(eps=0)).simplify_full()
print("first order of G_tt equals 2 Laplacian(Phi):", bool((lin(Gtt) - 2*lap).simplify_full() == 0))
print("first order of R_tt equals Laplacian(Phi):", bool((lin(Ric[0, 0].expr()) - lap).simplify_full() == 0))
