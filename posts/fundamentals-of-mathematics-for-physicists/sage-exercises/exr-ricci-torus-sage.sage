# Exercise exr-ricci-torus-sage: the Ricci tensor of the torus metric is K g with K = cos(a) / (r (R + r cos a)), the
# Gauss curvature, as for every two-dimensional metric (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
R, r = var('R r', domain='real'); assume(R > r, r > 0)
M, X, g = chart_metric('a b', lambda c: [[r**2, 0], [0, (R + r*cos(c[0]))**2]], ranges='a:(0,2*pi) b:(0,2*pi)')
a, b = X[:]
Ric = g.ricci()
print("Ricci tensor:", Ric.display())
K = cos(a)/(r*(R + r*cos(a)))
print("R_aa = r^2 K:", bool((Ric[0, 0].expr() - r**2*K).simplify_full() == 0), "; R_bb = (R + r cos a)^2 K:", bool((Ric[1, 1].expr() - (R + r*cos(a))**2*K).simplify_full() == 0), "; R_ab = 0:", bool(Ric[0, 1].expr() == 0))
print("Ric = K g:", Ric == K*g)
print("Ric is not a constant multiple of g: K at a = 0 is", K.subs(a=0).factor(), "and at a = pi it is", K.subs(a=pi).factor())
