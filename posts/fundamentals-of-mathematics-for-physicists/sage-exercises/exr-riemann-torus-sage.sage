# Exercise exr-riemann-torus-sage: the Riemann tensor of the torus metric with SageManifolds; the single independent
# component is R^a_bab = cos(a) (R + r cos a) / r.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
R, r = var('R r', domain='real'); assume(R > r, r > 0)
M, X, g = chart_metric('a b', lambda c: [[r**2, 0], [0, (R + r*cos(c[0]))**2]], ranges='a:(0,2*pi) b:(0,2*pi)')
a, b = X[:]
Rm = g.riemann()
print("Riemann tensor:", Rm.display())
print("R^a_bab =", Rm[0, 1, 0, 1].expr().factor(), ":", bool((Rm[0, 1, 0, 1].expr() - cos(a)*(R + r*cos(a))/r).simplify_full() == 0))
print("R^b_aab =", Rm[1, 0, 0, 1].expr().factor(), "= -g_aa R^a_bab / g_bb:", bool((Rm[1, 0, 0, 1].expr() + r**2*Rm[0, 1, 0, 1].expr()/(R + r*cos(a))**2).simplify_full() == 0))
nonzero = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2) if Rm[i, j, k, l].expr() != 0]
print("non-zero components (rho, sigma, mu, nu):", nonzero)
print("vanishes on the top and bottom circles a = pi/2, 3pi/2:", bool(Rm[0, 1, 0, 1].expr().subs(a=pi/2) == 0 and Rm[0, 1, 0, 1].expr().subs(a=3*pi/2) == 0))
