# Exercise exr-hyp-disc-geodesics: the Cayley map w = (z - i)/(z + i) sends the half-plane geodesic (x - 1)^2 + y^2 = 1
# to an arc of the circle |w + 1 + 2i| = 2, which meets the unit circle at right angles; a diameter of the disc,
# traversed with u = tanh(s/2), solves the geodesic equations of the disc metric.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
s = var('s', domain='real')
z = 1 + tanh(s) + I/cosh(s)                       # unit-speed semicircle with a = 1, R = 1
w = (z - I)/(z + I)
u, v = w.real().simplify_full(), w.imag().simplify_full()
c0 = -1 - 2*I                                     # centre, radius 2
print("|w + 1 + 2i|^2 =", ((u + 1)**2 + (v + 2)**2).simplify_full())
print("orthogonal to the unit circle, |c|^2 = 1 + rho^2:", abs(c0)**2, "= 1 +", 2**2)
print("end points: s -> -oo gives", (limit(u, s=-oo), limit(v, s=-oo)), "; s -> +oo gives", (limit(u, s=oo), limit(v, s=oo)))
M, X, g = chart_metric('p q', lambda c: [[4/(1 - c[0]**2 - c[1]**2)**2, 0], [0, 4/(1 - c[0]**2 - c[1]**2)**2]])
G = g.christoffel_symbols()
pu, pv = tanh(s/2), 0*s
vel = [diff(pu, s), diff(pv, s)]
eqs = [(diff([pu, pv][k], s, 2) + sum(G[k, i, j].expr().subs({X[0]: pu, X[1]: pv})*vel[i]*vel[j] for i in range(2) for j in range(2))).simplify_full() for k in range(2)]
print("diameter u = tanh(s/2), v = 0: geodesic equations give", eqs, "; speed^2 =", (4*vel[0]**2/(1 - pu**2)**2).simplify_full())
# the image of the semicircle solves the disc geodesic equations as well
vel = [diff(u, s), diff(v, s)]
eqs = [(diff([u, v][k], s, 2) + sum(G[k, i, j].expr().subs({X[0]: u, X[1]: v})*vel[i]*vel[j] for i in range(2) for j in range(2))).simplify_full() for k in range(2)]
print("image of the semicircle: geodesic equations give", eqs)
