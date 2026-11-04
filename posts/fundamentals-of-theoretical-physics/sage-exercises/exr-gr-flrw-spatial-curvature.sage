# Exercise exr-gr-flrw-spatial-curvature: the spatial metric of FLRW, dr^2/(1 - k r^2) + r^2 dOmega^2 (positive definite),
# has R_ij = 2 k gamma_ij and scalar curvature 6 k: a space of constant curvature k (a three-sphere of radius 1/sqrt(k) for
# k > 0, flat for k = 0, hyperbolic for k < 0).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
k = var('k')
M, X, g = chart_metric('r th ph', lambda c: [[1/(1 - k*c[0]^2), 0, 0], [0, c[0]^2, 0], [0, 0, c[0]^2*sin(c[1])^2]], ranges='r:(0,+oo) th:(0,pi) ph:(0,2*pi)')
Ric, Rs = g.ricci(), g.ricci_scalar()
print("R_ij = 2k gamma_ij:", Ric - 2*k*g == 0, "; scalar curvature:", Rs.expr().simplify_full())
