# Exercise exr-gauss-negative-sage: surfaces of revolution du^2 + f(u)^2 dphi^2 with K = -f''/f: f = e^{-u} (pseudosphere)
# and f = cosh u both have constant Gauss curvature -1, f = u sin(alpha) (cone) has K = 0 (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
alpha = var('alpha', domain='real'); assume(0 < alpha, alpha < pi/2)
for name, f in [("pseudosphere f = e^{-u}", lambda u: exp(-u)), ("f = cosh u", lambda u: cosh(u)), ("cone f = u sin(alpha)", lambda u: u*sin(alpha)), ("sphere f = sin u", lambda u: sin(u))]:
    M, X, g = chart_metric('u ph', lambda c: [[1, 0], [0, f(c[0])**2]], ranges='u:(0,+oo) ph:(0,2*pi)')
    u, ph = X[:]
    K = (g.ricci_scalar().expr()/2).simplify_full()
    print("%s: K = %s ; equals -f''/f: %s" % (name, K, bool((K + diff(f(u), u, 2)/f(u)).simplify_full() == 0)))
