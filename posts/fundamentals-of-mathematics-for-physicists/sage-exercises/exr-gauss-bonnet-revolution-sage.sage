# Exercise exr-gauss-bonnet-revolution-sage: for a surface of revolution du^2 + f(u)^2 dphi^2 the total curvature over
# u0 <= u <= u1 is -2 pi [f'(u1) - f'(u0)] (f > 0): 4 pi for the sphere, 0 for the torus, -4 pi sinh L for f = cosh u.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
u = var('u', domain='real')
F = function('f')(u)
M, X, g = chart_metric('u ph', lambda c: [[1, 0], [0, function('f')(c[0])**2]], ranges='u ph:(0,2*pi)')
uu, ph = X[:]
K = (g.ricci_scalar().expr()/2).simplify_full()
dA = g.sqrt_abs_det().expr()
print("K =", K, "; with f > 0, sqrt(det g) = f and K f = -f''(u):", bool((K*function('f')(uu) + diff(function('f')(uu), uu, 2)).simplify_full() == 0))
print("so the integral of K dA over u0 <= u <= u1 is -2 pi [f'(u1) - f'(u0)]")
L = var('L', domain='real'); assume(L > 0)
cases = [("sphere, f = sin u on [0, pi]", sin(u), 0, pi), ("torus, f = R + r cos u on [0, 2 pi] (r = 1)", 2 + cos(u), 0, 2*pi), ("f = cosh u (K = -1) on [-L, L]", cosh(u), -L, L)]
for name, f, u0, u1 in cases:
    total = (-2*pi*(diff(f, u).subs(u=u1) - diff(f, u).subs(u=u0))).simplify_full()
    direct = integrate(integrate(-diff(f, u, 2), ph, 0, 2*pi), u, u0, u1).simplify_full()
    print("%s: total curvature = %s ; direct integration agrees: %s" % (name, total, bool((total - direct).simplify_full() == 0)))
print("for f = cosh u the total curvature -4 pi sinh L grows without bound: the surface is not compact")
