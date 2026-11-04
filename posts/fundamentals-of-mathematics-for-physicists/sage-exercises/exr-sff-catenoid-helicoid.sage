# Exercise exr-sff-catenoid-helicoid: the helicoid (rho cos ph, rho sin ph, c ph) and the catenoid
# (sqrt(c^2 + rho^2) cos ph, sqrt(c^2 + rho^2) sin ph, c asinh(rho/c)) have the same induced metric
# d rho^2 + (c^2 + rho^2) d ph^2 but different second fundamental forms; both have K = -c^2/(c^2 + rho^2)^2, which the
# intrinsic formula -f''/f confirms.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _surfaces import *
rho, ph, c = var('rho ph c', domain='real'); assume(c > 0)
for name, E in [("helicoid", [rho*cos(ph), rho*sin(ph), c*ph]),
                ("catenoid", [sqrt(c**2 + rho**2)*cos(ph), sqrt(c**2 + rho**2)*sin(ph), c*arcsinh(rho/c)])]:
    g, n, h, S, K = second_form(E, rho, ph)
    print(name, ": g =", g.list(), "; h =", h.list(), "; principal curvatures", sorted([e.simplify_full() for e in S.eigenvalues()], key=str), "; K =", K.factor())
f = sqrt(c**2 + rho**2)
print("intrinsic: K = -f''/f =", (-diff(f, rho, 2)/f).simplify_full().factor())
