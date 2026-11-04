# Exercise exr-gr-nl-exact-christoffel: for g_tt = 1 + 2 Phi, g_ij = -(1 - 2 Phi) delta_ij with an arbitrary static
# Phi(x, y, z), the exact Christoffel symbols are Gamma^i_tt = d_i Phi/(1 - 2 Phi) and Gamma^t_ti = d_i Phi/(1 + 2 Phi);
# to first order both are d_i Phi.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M = Manifold(4, 'M', structure='Lorentzian'); X = M.chart('t x y z'); t, x, y, z = X[:]
Phi = function('Phi')(x, y, z)
g = M.metric('g'); g[0, 0] = 1 + 2*Phi
for i in (1, 2, 3): g[i, i] = -(1 - 2*Phi)
Gm = g.christoffel_symbols()
ok1 = all(bool((Gm[i, 0, 0].expr() - diff(Phi, c)/(1 - 2*Phi)).simplify_full() == 0) for i, c in zip((1, 2, 3), (x, y, z)))
ok2 = all(bool((Gm[0, 0, i].expr() - diff(Phi, c)/(1 + 2*Phi)).simplify_full() == 0) for i, c in zip((1, 2, 3), (x, y, z)))
print("Gamma^i_tt = d_i Phi/(1 - 2 Phi):", ok1, "; Gamma^t_ti = d_i Phi/(1 + 2 Phi):", ok2)
print("Gamma^x_yy =", Gm[1, 2, 2].expr().simplify_full(), " (multiplies v^2: negligible for slow particles)")
