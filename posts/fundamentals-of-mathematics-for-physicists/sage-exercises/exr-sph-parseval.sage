# Exercise exr-sph-parseval: Parseval on the sphere for f = cos theta + sin theta cos phi + sin^2 theta cos 2phi: the
# coefficients f_lm with l <= 2 and the identity int |f|^2 = sum |f_lm|^2.
th, ph = var("theta phi")
Y = lambda l, m: spherical_harmonic(l, m, th, ph)
f = cos(th) + sin(th) * cos(ph) + sin(th) ** 2 * cos(2 * ph)
def coeff(l, m):
    return integrate(integrate((conjugate(Y(l, m)) * f).simplify_full() * sin(th), ph, 0, 2 * pi), th, 0, pi).simplify_full()
cs = {(l, m): coeff(l, m) for l in range(3) for m in range(-l, l + 1)}
nz = {key: v for key, v in cs.items() if v != 0}
print("non-zero coefficients:", nz)
norm = integrate(integrate((f ** 2).expand() * sin(th), ph, 0, 2 * pi), th, 0, pi)
print("int |f|^2 dOmega =", norm, "; sum |f_lm|^2 =", sum(v ** 2 for v in nz.values()).simplify_full(), "; equal:", bool((norm - sum(v ** 2 for v in nz.values())).simplify_full() == 0))
