# Exercise exr-hyp-moebius: a real Moebius map z -> (a z + b)/(c z + d) with a d - b c = 1 preserves the upper
# half-plane and the metric |dz|^2 / y^2 (SageManifolds pullback with a general matrix, d eliminated).
from sage.all import *
H = Manifold(2, 'H', structure='Riemannian'); X = H.chart('x y:(0,+oo)'); x, y = X[:]
g = H.metric('g'); g[0, 0] = 1/y**2; g[1, 1] = 1/y**2
a, b, c = var('a b c', domain='real'); assume(a != 0)
d = (1 + b*c)/a                                 # a d - b c = 1
z = x + I*y
w = ((a*z + b)/(c*z + d))
u = w.real().simplify_full(); v = w.imag().simplify_full()
print("Im w =", v.factor(), "= y / |c z + d|^2:", bool((v - y/((c*x + d)**2 + c**2*y**2)).simplify_full() == 0))
F = H.diff_map(H, {(X, X): [u, v]}, name='F')
pb = F.pullback(g)
print("pullback of g equals g:", all(bool((pb[i, j].expr() - g[i, j].expr()).simplify_full() == 0) for i in range(2) for j in range(2)))
print("translations (a, b, c) = (1, b, 0) and dilations (a, 0, 0) are special cases; the inversion has a = 0:")
Finv = H.diff_map(H, {(X, X): [-x/(x**2 + y**2), y/(x**2 + y**2)]}, name='Inv')
pbi = Finv.pullback(g)
print("z -> -1/z preserves g:", all(bool((pbi[i, j].expr() - g[i, j].expr()).simplify_full() == 0) for i in range(2) for j in range(2)))
