# Exercise exr-torus-curvature: the scalar curvature of the torus changes sign and integrates to zero (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
R, r = var('R r', domain='real'); assume(R > r, r > 0)
M, X, g = chart_metric('a b', lambda c: [[r**2, 0], [0, (R + r*cos(c[0]))**2]], ranges='a:(0,2*pi) b:(0,2*pi)')
Rm, Ric, Rs = curvature(g)
a, b = X[:]
Rs_expr = Rs.expr().simplify_full().factor()
print("scalar curvature:", Rs_expr)
print("outer equator a = 0:", Rs_expr.subs(a=0).factor(), "   inner equator a = pi:", Rs_expr.subs(a=pi).factor())
K = Rs_expr/2                                                          # Gauss curvature K = R/2 in two dimensions
print("integral of K dA over the torus:", integrate(integrate(K*g.sqrt_abs_det().expr(), b, 0, 2*pi), a, 0, 2*pi).simplify_full())
