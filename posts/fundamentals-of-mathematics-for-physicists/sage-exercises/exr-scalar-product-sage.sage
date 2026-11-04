# Exercise exr-scalar-product-sage: the scalar curvature of the product of two round spheres of radii p and q is additive,
# R = 2/p^2 + 2/q^2 (SageManifolds, four-dimensional product metric).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
p, q = var('p q', domain='real'); assume(p > 0, q > 0)
M, X, g = chart_metric('th1 ph1 th2 ph2', lambda c: [[p**2, 0, 0, 0], [0, p**2*sin(c[0])**2, 0, 0], [0, 0, q**2, 0], [0, 0, 0, q**2*sin(c[2])**2]],
                       ranges='th1:(0,pi) ph1:(0,2*pi) th2:(0,pi) ph2:(0,2*pi)')
Rs = g.ricci_scalar().expr().simplify_full()
print("scalar curvature of S^2(p) x S^2(q):", Rs, ":", bool(Rs == 2/p**2 + 2/q**2))
Ric = g.ricci()
print("Ricci tensor:", Ric.display())
print("mixed components R_{th1 th2}, R_{ph1 ph2} vanish:", bool(Ric[0, 2].expr() == 0 and Ric[1, 3].expr() == 0))
print("for p = q = 1: R = 4 = 2 + 2, twice the value of one unit sphere:", bool(Rs.subs(p=1, q=1) == 4))
