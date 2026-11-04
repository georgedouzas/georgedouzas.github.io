# Exercise exr-sphere-riemann-sage: all sixteen components R^rho_sigma mu nu of the unit sphere with SageManifolds; the
# four with (mu, nu) = (theta, phi) are those of the proposition and the other twelve follow by antisymmetry.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
th, ph = X[:]
Rm = g.riemann()
names = ['th', 'ph']
for a in range(2):
    for b in range(2):
        print("  R^%s_%s %s %s = %s   R^%s_%s %s %s = %s" % (names[a], names[b], 'th', 'ph', Rm[a, b, 0, 1].expr(), names[a], names[b], 'ph', 'th', Rm[a, b, 1, 0].expr()))
print("R^th_ph th ph = sin^2 th, R^ph_th th ph = -1, R^th_th th ph = 0, R^ph_ph th ph = 0:", bool(Rm[0, 1, 0, 1].expr() == sin(th)**2 and Rm[1, 0, 0, 1].expr() == -1 and Rm[0, 0, 0, 1].expr() == 0 and Rm[1, 1, 0, 1].expr() == 0))
print("components with mu = nu vanish:", all(bool(Rm[a, b, c, c].expr() == 0) for a in range(2) for b in range(2) for c in range(2)))
print("number of non-zero components:", sum(1 for a in range(2) for b in range(2) for c in range(2) for d in range(2) if Rm[a, b, c, d].expr() != 0), "(the two independent ones and their mirror images)")
Rd = Rm.down(g)
print("lowered: R_th ph th ph = sin^2 th = det g:", bool((Rd[0, 1, 0, 1].expr() - g.determinant().expr()).simplify_full() == 0))
