# Exercise exr-cc-surface-pipeline: the whole pipeline metric -> Christoffel symbols -> Riemann -> Ricci -> scalar curvature
# -> total curvature for the torus of revolution with R = 2, r = 1, starting from its embedding in R^3 (SageManifolds).
from sage.all import *
E3 = Manifold(3, 'R^3', structure='Riemannian'); C = E3.chart('X Y Z')
h = E3.metric('h'); h[0, 0] = 1; h[1, 1] = 1; h[2, 2] = 1
R, r = 2, 1
T2 = Manifold(2, 'T^2', structure='Riemannian'); ab = T2.chart('a:(0,2*pi) b:(0,2*pi)'); a, b = ab[:]
emb = T2.diff_map(E3, {(ab, C): [(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)]}, name='E')
g = T2.metric('g'); g.set(emb.pullback(h))
print("1. induced metric:", g.display())
print("2. Christoffel symbols:"); print(g.christoffel_symbols_display())
Rm = g.riemann()
print("3. Riemann tensor: R^a_bab =", Rm[0, 1, 0, 1].expr().factor())
Ric = g.ricci()
print("4. Ricci tensor:", Ric.display())
Rs = g.ricci_scalar().expr().simplify_full()
print("5. scalar curvature R =", Rs.factor(), "; Gauss curvature K = R/2")
K = Rs/2; dA = g.sqrt_abs_det().expr().simplify_full()
print("   K on the outer equator:", K.subs(a=0), "; inner equator:", K.subs(a=pi), "; top circle:", K.subs(a=pi/2))
print("6. area element sqrt(det g) =", dA, "; area =", integrate(integrate(dA, a, 0, 2*pi), b, 0, 2*pi))
print("   total curvature, integral of K dA =", integrate(integrate((K*dA).simplify_full(), a, 0, 2*pi), b, 0, 2*pi), "= 2 pi chi(T^2) with chi = 0")
print("   positive part (outer half):", integrate(integrate((K*dA).simplify_full(), a, -pi/2, pi/2), b, 0, 2*pi), "; negative part (inner half):", integrate(integrate((K*dA).simplify_full(), a, pi/2, 3*pi/2), b, 0, 2*pi))
print("7. Einstein tensor G = Ric - R g / 2 vanishes (two dimensions):", Ric - g.ricci_scalar()*g/2 == 0)
