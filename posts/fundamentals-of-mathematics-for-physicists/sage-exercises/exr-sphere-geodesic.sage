# Exercise exr-sphere-geodesic: a tilted great circle satisfies the geodesic equations of the round sphere
# (SageManifolds: the Christoffel symbols come from the metric, the equations are assembled from them).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
Gam = g.christoffel_symbols(); th, ph = X[:]
t, alpha = var('t alpha', domain='real'); assume(0 < alpha, alpha < pi/2)
P = (cos(t), cos(alpha)*sin(t), sin(alpha)*sin(t))                   # the great circle tilted by alpha, in R^3
th_t = arccos(P[2]); ph_t = atan2(P[1], P[0])                        # its spherical coordinates
sub = {th: th_t, ph: ph_t}
coords = [th_t, ph_t]
residuals = []
for k in range(2):
    r = diff(coords[k], t, 2) + sum(Gam[k, i, j].expr().subs(sub)*diff(coords[i], t)*diff(coords[j], t) for i in range(2) for j in range(2))
    residuals.append(r.simplify_full().trig_simplify())
print("theta-equation residual:", residuals[0]); print("phi-equation residual:", residuals[1])
