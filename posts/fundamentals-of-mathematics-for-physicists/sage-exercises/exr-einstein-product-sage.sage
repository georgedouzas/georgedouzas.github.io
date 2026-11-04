# Exercise exr-einstein-product-sage: the product S^2(p) x S^2(q) of round spheres is an Einstein metric exactly when the
# radii agree: the Ricci tensor is (1/p^2) g on the first factor and (1/q^2) g on the second (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
p, q = var('p q', domain='real'); assume(p > 0, q > 0)
M, X, g = chart_metric('th1 ph1 th2 ph2', lambda c: [[p**2, 0, 0, 0], [0, p**2*sin(c[0])**2, 0, 0], [0, 0, q**2, 0], [0, 0, 0, q**2*sin(c[2])**2]],
                       ranges='th1:(0,pi) ph1:(0,2*pi) th2:(0,pi) ph2:(0,2*pi)')
Ric = g.ricci()
ratios = [(Ric[i, i].expr()/g[i, i].expr()).simplify_full() for i in range(4)]
print("R_ii / g_ii for the four coordinates:", ratios)
print("first factor: Ric = g/p^2; second factor: Ric = g/q^2:", bool(ratios[0] == 1/p**2 and ratios[1] == 1/p**2 and ratios[2] == 1/q**2 and ratios[3] == 1/q**2))
print("Einstein for p = q = 1 with lambda = 1:", all(bool((Ric[i, j].expr() - g[i, j].expr()).subs(p=1, q=1) == 0) for i in range(4) for j in range(4)))
print("not Einstein for p = 1, q = 2: Ric - g has the component", (Ric[2, 2].expr() - g[2, 2].expr()).subs(p=1, q=2))
lam = var('lam')
print("solving R_ii = lam g_ii for all i forces 1/p^2 = 1/q^2 = lam:", solve([ratios[0] == lam, ratios[2] == lam], [lam, q])[0])
