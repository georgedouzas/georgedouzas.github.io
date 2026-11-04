# Exercise exr-riemann-bianchi-sage: symmetries of the lowered Riemann tensor of the SU(2) metric: antisymmetry in both
# pairs, pair symmetry and the first Bianchi identity, leaving n^2 (n^2 - 1)/12 = 6 independent components (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph ps', lambda c: [[QQ(1)/4, 0, 0], [0, QQ(1)/4, cos(c[0])/4], [0, cos(c[0])/4, QQ(1)/4]], ranges='th:(0,pi) ph:(0,2*pi) ps:(0,4*pi)')
Rd = g.riemann().down(g)                                     # R_{rho sigma mu nu}
n = 3; I = range(n)
val = lambda a, b, c, d: Rd[a, b, c, d].expr()
print("antisymmetric in the last pair:", all(bool((val(a, b, c, d) + val(a, b, d, c)).simplify_full() == 0) for a in I for b in I for c in I for d in I))
print("antisymmetric in the first pair:", all(bool((val(a, b, c, d) + val(b, a, c, d)).simplify_full() == 0) for a in I for b in I for c in I for d in I))
print("symmetric under pair exchange:", all(bool((val(a, b, c, d) - val(c, d, a, b)).simplify_full() == 0) for a in I for b in I for c in I for d in I))
print("first Bianchi identity:", all(bool((val(a, b, c, d) + val(a, c, d, b) + val(a, d, b, c)).simplify_full() == 0) for a in I for b in I for c in I for d in I))
pairs = [(a, b) for a in I for b in I if a < b]
indep = [(p, q) for p in pairs for q in pairs if pairs.index(p) <= pairs.index(q)]
print("pairs:", len(pairs), "; symmetric matrix on pairs:", len(indep), "entries; Bianchi conditions:", binomial(n, 4), "; independent components:", len(indep) - binomial(n, 4), "= n^2 (n^2 - 1)/12 =", n**2*(n**2 - 1)/12)
print("R_th,ph,th,ph =", val(0, 1, 0, 1).simplify_full(), "= g_thth g_phph - g_thph^2 (sectional curvature 1):", bool((val(0, 1, 0, 1) - (g[0, 0].expr()*g[1, 1].expr() - g[0, 1].expr()**2)).simplify_full() == 0))
