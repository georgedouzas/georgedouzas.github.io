# Exercise exr-christoffel-symm-sage: for a metric with arbitrary component functions on the plane the Christoffel symbols
# are symmetric in the lower indices, and there are n^2 (n+1)/2 = 6 independent ones (SageManifolds).
from sage.all import *
M = Manifold(2, 'M', structure='Riemannian'); X = M.chart('x y'); x, y = X[:]
E = function('E')(x, y); F = function('F')(x, y); G = function('G')(x, y)   # Gauss's notation for a general metric
g = M.metric('g'); g[0, 0] = E; g[0, 1] = F; g[1, 1] = G
Gam = g.christoffel_symbols()
print("Gamma^x_xx =", Gam[0, 0, 0].expr().factor())
print("symmetric in the lower indices:", all(bool((Gam[k, i, j].expr() - Gam[k, j, i].expr()).simplify_full() == 0) for k in range(2) for i in range(2) for j in range(2)))
indep = [(k, i, j) for k in range(2) for i in range(2) for j in range(i, 2)]
print("independent symbols:", len(indep), "= n^2 (n+1)/2 with n = 2:", len(indep) == 2**2*3/2)
h = M.metric('h'); h[0, 0] = E; h[1, 1] = G                   # an orthogonal chart, F = 0
Gh = h.christoffel_symbols()
print("orthogonal chart (F = 0): Gamma^x_yy =", Gh[0, 1, 1].expr(), "and Gamma^y_xy =", Gh[1, 0, 1].expr())
