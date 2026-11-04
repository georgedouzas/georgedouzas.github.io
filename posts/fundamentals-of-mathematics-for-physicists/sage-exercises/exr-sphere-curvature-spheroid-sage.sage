# Exercise exr-sphere-curvature-spheroid-sage: the spheroid x^2 + y^2 + z^2/c^2 = 1 has non-constant scalar curvature,
# largest at the poles for c > 1 (prolate) and at the equator for c < 1 (oblate); c = 1 gives back R = 2 (SageManifolds).
from sage.all import *
E3 = Manifold(3, 'R^3', structure='Riemannian'); C = E3.chart('X Y Z')
h = E3.metric('h'); h[0, 0] = 1; h[1, 1] = 1; h[2, 2] = 1
c = var('c', domain='real'); assume(c > 0)
S = Manifold(2, 'S', structure='Riemannian'); A = S.open_subset('A'); X = A.chart(r'th:(0,pi):\theta ph:(0,2*pi):\varphi'); th, ph = X[:]
emb = A.diff_map(E3, {(X, C): [sin(th)*cos(ph), sin(th)*sin(ph), c*cos(th)]}, name='E')
g = A.metric('g'); g.set(emb.pullback(h))
print("induced metric:", g.display())
Rs = g.ricci_scalar().expr().simplify_full().factor()
print("scalar curvature R(theta) =", Rs)
print("R = 2 for c = 1:", bool(Rs.subs(c=1).simplify_full() == 2))
print("poles (theta -> 0): R =", limit(Rs, th=0).factor(), "; equator (theta = pi/2): R =", Rs.subs(th=pi/2).factor())
print("prolate c = 2: R(pole) = %.3f > R(equator) = %.3f:" % (float(limit(Rs, th=0).subs(c=2)), float(Rs.subs(th=pi/2, c=2))), bool(limit(Rs, th=0).subs(c=2) > Rs.subs(th=pi/2, c=2)))
print("oblate c = 1/2: R(pole) = %.3f < R(equator) = %.3f:" % (float(limit(Rs, th=0).subs(c=1/2)), float(Rs.subs(th=pi/2, c=1/2))), bool(limit(Rs, th=0).subs(c=QQ(1)/2) < Rs.subs(th=pi/2, c=QQ(1)/2)))
K = Rs/2; dA = g.sqrt_abs_det().expr()
print("Gauss-Bonnet for c = 2: integral of K dA =", integrate(integrate((K*dA).subs(c=2).simplify_full(), ph, 0, 2*pi), th, 0, pi).simplify_full())
