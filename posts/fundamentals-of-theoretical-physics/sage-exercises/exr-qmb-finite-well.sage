# Exercise exr-qmb-finite-well (Cini et al., Sec. 8.7, Problem 3, and Sec. 9.8, Problem 2): bound states of the finite
# square well V = -V0 for |x| < a (units hbar^2/2m = 1, a = 1, z0 = sqrt(V0)). Even states solve z tan z = sqrt(z0^2 - z^2),
# odd states -z cot z = sqrt(z0^2 - z^2); their number is ceil(2 z0/pi), compared with the count found numerically.
def levels(z0):
    zs = []
    f_even = lambda z: z * tan(z) - sqrt(z0^2 - z^2)
    f_odd = lambda z: -z / tan(z) - sqrt(z0^2 - z^2)
    n = 0
    while n * pi / 2 < z0:
        lo, hi = n * pi / 2 + 1e-9, min((n + 1) * pi / 2 - 1e-9, z0 - 1e-12)
        f = f_even if n % 2 == 0 else f_odd
        if lo < hi and f(lo) * f(hi) < 0:
            zs.append(find_root(f, lo, hi))
        n += 1
    return zs
for z0 in (0.5, 1.6, 3.0, 4.8, 10.0):
    zs = levels(z0)
    print("z0 = %4.1f: %d bound states (ceil(2 z0/pi) = %d), energies E/V0 + 1 = %s" % (z0, len(zs), ceil(2 * z0 / pi.n()), [round((z / z0)^2, 4) for z in zs]))
