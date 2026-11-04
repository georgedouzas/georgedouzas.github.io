# Exercise exr-circle-christoffel-sage: the circle has Gamma = 0 in the angle chart, Gamma^u_uu = 1/u in the chart phi = u^2,
# and no curvature in either (SageManifolds, one-dimensional manifold).
from sage.all import *
S1 = Manifold(1, 'S^1', structure='Riemannian')
A = S1.chart(r'ph:(0,2*pi):\varphi'); ph = A[0]
g = S1.metric('g'); g[0, 0] = 1
print("angle chart: Gamma^ph_phph =", g.christoffel_symbols()[0, 0, 0].expr(), "-> zero:", bool(g.christoffel_symbols()[0, 0, 0].expr() == 0))
U = S1.open_subset('U'); B = U.chart('u:(0,sqrt(2*pi))'); u = B[0]
tr = B.transition_map(A.restrict(U), [u**2]); tr.set_inverse(sqrt(ph), check=False)
gU = g.restrict(U)
print("in the chart phi = u^2 the metric is", gU.display(B.frame(), B))
GamU = gU.christoffel_symbols(B)
print("Gamma^u_uu =", GamU[0, 0, 0].expr(), ":", bool(GamU[0, 0, 0].expr() == 1/u))
print("Riemann tensor in the u-chart:", gU.riemann().display(B.frame(), B), "; scalar curvature:", gU.ricci_scalar().expr(B))
