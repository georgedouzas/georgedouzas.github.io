# Exercise exr-riemann-antisymm-sage: for a metric with arbitrary component functions on the plane the Riemann tensor is
# antisymmetric in its last pair and vanishes when the last two indices coincide (SageManifolds).
from sage.all import *
M = Manifold(2, 'M', structure='Riemannian'); X = M.chart('x y'); x, y = X[:]
E = function('E')(x, y); F = function('F')(x, y); G = function('G')(x, y)
g = M.metric('g'); g[0, 0] = E; g[0, 1] = F; g[1, 1] = G
Rm = g.riemann()
I = range(2)
print("R^rho_sigma nu mu = -R^rho_sigma mu nu:", all(bool((Rm[a, b, c, d].expr() + Rm[a, b, d, c].expr()).simplify_full() == 0) for a in I for b in I for c in I for d in I))
print("R^rho_sigma mu mu = 0:", all(bool(Rm[a, b, c, c].expr().simplify_full() == 0) for a in I for b in I for c in I))
Rd = Rm.down(g)
print("lowered tensor antisymmetric in the first pair (needs metric compatibility):", all(bool((Rd[a, b, c, d].expr() + Rd[b, a, c, d].expr()).simplify_full() == 0) for a in I for b in I for c in I for d in I))
h = M.metric('h'); h[0, 0] = 1; h[1, 1] = function('f')(x)**2    # orthogonal chart with E = 1, G = f(x)^2
K = (h.riemann().down(h)[0, 1, 0, 1].expr()/h.determinant().expr()).simplify_full()
print("orthogonal chart dx^2 + f(x)^2 dy^2: K = R_xyxy / det g =", K, "= -f''/f:", bool((K + diff(function('f')(x), x, 2)/function('f')(x)).simplify_full() == 0))
