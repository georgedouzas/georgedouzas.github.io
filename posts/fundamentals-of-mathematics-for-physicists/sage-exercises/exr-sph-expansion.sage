# Exercise exr-sph-expansion: expansion of sin^2 theta cos 2phi and cos^2 theta in spherical harmonics with l <= 2, by
# exact integration of the coefficients f_lm = int conj(Y_lm) f dOmega.
th, ph = var("theta phi")
Y = lambda l, m: spherical_harmonic(l, m, th, ph)
def coeff(f, l, m):
    return integrate(integrate((conjugate(Y(l, m)) * f).simplify_full() * sin(th), ph, 0, 2 * pi), th, 0, pi).simplify_full()
for name, f in [("sin^2 theta cos 2phi", sin(th) ** 2 * cos(2 * ph)), ("cos^2 theta", cos(th) ** 2)]:
    cs = {(l, m): coeff(f, l, m) for l in range(3) for m in range(-l, l + 1)}
    nz = {key: v for key, v in cs.items() if v != 0}
    print(name, ": non-zero f_lm =", nz)
    rec = sum(v * Y(l, m) for (l, m), v in nz.items())
    print("  expansion reproduces the function:", bool((rec - f).simplify_full().trig_reduce() == 0) or bool(abs((rec - f).subs({th: 0.7, ph: 1.3}).n()) < 1e-12))
