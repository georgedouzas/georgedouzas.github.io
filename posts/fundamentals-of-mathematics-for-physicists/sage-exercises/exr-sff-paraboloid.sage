# Exercise exr-sff-paraboloid: on the paraboloid z = x^2 + y^2 in polar coordinates the Riemann component of the
# induced metric, R_{rho ph rho ph} = 4 rho^2/(1 + 4 rho^2), equals det h, as Gauss's equation requires, and
# K = 4/(1 + 4 rho^2)^2.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _surfaces import *; from _curvature import *
rho, ph = var('rho ph', domain='real'); assume(rho > 0)
g, n, h, S, K = second_form([rho*cos(ph), rho*sin(ph), rho**2], rho, ph)
h = h.apply_map(lambda e: e.canonicalize_radical().simplify_full())
print("induced metric:", g.list(), "; h =", h.list(), "; principal curvatures", [e.canonicalize_radical().simplify_full() for e in S.diagonal()])
M, X, gm = chart_metric('r p', lambda co: [[1 + 4*co[0]**2, 0], [0, co[0]**2]], ranges='r:(0,+oo) p')
R1212 = gm.riemann().down(gm)[0, 1, 0, 1].expr().simplify_full()
print("intrinsic R_{rho ph rho ph} =", R1212, "; det h =", h.det().subs(rho=X[0]).simplify_full(), "; equal:", bool((R1212 - h.det().subs(rho=X[0])).simplify_full() == 0))
print("K = R_1212/det g =", (R1212/gm.determinant().expr()).factor(), "; extrinsic det h/det g =", K.factor())
