# Exercise exr-sff-cone-torus: second fundamental forms of the cone of half-angle alpha and of the torus, with the
# normal n = E_1 x E_2 / |E_1 x E_2|: the cone has h = rho sin(alpha) cos(alpha) dphi^2, the torus
# h = r da^2 + (R + r cos a) cos a db^2.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _surfaces import *
rho, ph, al = var('rho ph alpha', domain='real'); assume(rho > 0, al > 0, al < pi/2)
g, n, h, S, K = second_form([rho*sin(al)*cos(ph), rho*sin(al)*sin(ph), rho*cos(al)], rho, ph)
fix = lambda e: e.subs({abs(sin(al)): sin(al)}).simplify_full()
print("cone: g =", g.list(), "; n =", [fix(e) for e in n], "; h =", [fix(e) for e in h.list()], "; det h / det g =", K)
a, b, R, r = var('a b R r', domain='real'); assume(r > 0, R > r)
g, n, h, S, K = second_form([(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)], a, b)
h = h.apply_map(lambda e: e.canonicalize_radical().simplify_full())
print("torus: g =", g.list(), "; n =", list(n.apply_map(lambda e: e.canonicalize_radical().simplify_full())))
print("torus: h =", h.list())
print("h_bb = (R + r cos a) cos a:", bool((h[1, 1] - (R + r*cos(a))*cos(a)).simplify_full() == 0), "; h_aa = r:", bool(h[0, 0] == r))
