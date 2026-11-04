# Exercise exr-sff-pseudosphere: Beltrami's pseudosphere E = (sin s cos ph, sin s sin ph, cos s + log tan(s/2)),
# 0 < s < pi/2, has h = -cot(s) ds^2 + sin(s) cos(s) dph^2, principal curvatures -tan(s) and cot(s), and K = -1.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _surfaces import *
s, ph = var('s ph', domain='real'); assume(s > 0, s < pi/2)
# log tan(s/2) = log sin s - log(1 + cos s), written without half angles
g, n, h, S, K = second_form([sin(s)*cos(ph), sin(s)*sin(ph), cos(s) + log(sin(s)) - log(1 + cos(s))], s, ph)
simp = lambda e: e.simplify_trig().canonicalize_radical().simplify_trig().subs({abs(cos(s)): cos(s)}).simplify_trig()   # cos s > 0
g, h, S = g.apply_map(simp), h.apply_map(simp), S.apply_map(simp)
print("induced metric g =", g.list(), "= cot^2 s ds^2 + sin^2 s dph^2:", bool((g[0, 0] - cot(s)**2).simplify_trig() == 0) and bool(g[1, 1] == sin(s)**2))
n = n.apply_map(simp)
print("unit normal n =", list(n))
print("h =", h.list(), ": h_ss = -cot s:", bool((h[0, 0] + cot(s)).simplify_trig() == 0), ", h_phph = sin s cos s:", bool((h[1, 1] - sin(s)*cos(s)).simplify_trig() == 0))
print("shape operator S =", S.list(), ": principal curvatures -tan s and cot s:", bool((S[0, 0] + tan(s)).simplify_trig() == 0) and bool((S[1, 1] - cot(s)).simplify_trig() == 0))
print("K = det S =", simp(K))
print("at the rim s -> pi/2: n ->", [limit(e, s=pi/2) for e in n], ", -tan s ->", limit(-tan(s), s=pi/2, dir="-"), "(the cuspidal edge)")
