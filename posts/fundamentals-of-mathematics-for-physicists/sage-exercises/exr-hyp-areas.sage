# Exercise exr-hyp-areas: areas in the Poincare half-plane, with the volume element dx dy / y^2: the half-strip
# |x| < 1, y > 1 has area 2, the ideal triangle |x| < 1, x^2 + y^2 > 1 has area pi, and the region y < 1 has
# infinite area.
from sage.all import *
x, y = var('x y', domain='real')
dA = 1/y**2                                     # sqrt(det g) for g = (dx^2 + dy^2)/y^2
strip = integrate(integrate(dA, y, 1, oo), x, -1, 1)
print("area of the half-strip |x| < 1, y > 1:", strip)
assume(x > -1, x < 1)
inner = integrate(dA, y, sqrt(1 - x**2), oo)
print("inner integral over y > sqrt(1 - x^2):", inner.simplify_full())
tri = integrate(inner, x, -1, 1)
print("area of the ideal triangle with vertices -1, 1, oo:", tri, "; equal to pi - (0 + 0 + 0):", bool(tri == pi))
eps = var('eps'); assume(eps > 0, eps < 1)
low = integrate(integrate(dA, y, eps, 1), x, -1, 1)
print("area of |x| < 1, eps < y < 1:", low.simplify_full(), "-> oo as eps -> 0:", limit(low, eps=0, dir='+'))
