# Exercise exr-first-kind-su2-sage: Christoffel symbols of the first kind of the SU(2) metric in Euler angles, obtained by
# lowering the upper index, and the identity Gamma_kij + Gamma_ikj = d_j g_ik (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph ps', lambda c: [[QQ(1)/4, 0, 0], [0, QQ(1)/4, cos(c[0])/4], [0, cos(c[0])/4, QQ(1)/4]], ranges='th:(0,pi) ph:(0,2*pi) ps:(0,4*pi)')
th, ph, ps = X[:]
Gam = g.christoffel_symbols()
n = 3; names = ['th', 'ph', 'ps']
first = {}
for k in range(n):
    for i in range(n):
        for j in range(i, n):
            v = sum(g[k, m].expr()*Gam[m, i, j].expr() for m in range(n)).simplify_full()
            if v != 0: first[(names[k], names[i], names[j])] = v
print("non-zero first-kind symbols Gamma_kij (i <= j):", first)
print("Gamma_th,ph,ps = sin(th)/8:", bool(first[('th', 'ph', 'ps')] == sin(th)/8))
ok = all(bool((sum(g[k, m].expr()*Gam[m, i, j].expr() for m in range(n)) + sum(g[i, m].expr()*Gam[m, k, j].expr() for m in range(n)) - diff(g[i, k].expr(), X[j])).simplify_full() == 0) for k in range(n) for i in range(n) for j in range(n))
print("Gamma_kij + Gamma_ikj = d_j g_ik for all 27 index triples:", ok)
