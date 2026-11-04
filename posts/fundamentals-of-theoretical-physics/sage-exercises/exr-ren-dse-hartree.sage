# Exercise exr-ren-dse-hartree: exact <phi^2>, the Hartree solution and the two-loop truncation G^-1 = 1 + lam G/2 - lam^2 G^3/6.
import mpmath as mpm
def exact(l):
    f = lambda p: mpm.exp(-p**2 / 2 - l * p**4 / 24)
    return mpm.quad(lambda p: p**2 * f(p), [-mpm.inf, 0, mpm.inf]) / mpm.quad(f, [-mpm.inf, 0, mpm.inf])
G = polygen(RR, 'G')
for l in [0.5, 1, 10]:
    ex = float(exact(l)); h = float((sqrt(1 + 2 * l) - 1) / l)
    rts = [r for r in (-(l^2) / 6 * G^4 + l / 2 * G^2 + G - 1).roots(multiplicities=False) if 0 < r <= 1]
    tl = max(rts) if rts else None
    print("lam = %-4s exact %.4f  Hartree %.4f (%.1f%%)  two-loop %s" % (l, ex, h, 100 * abs(h - ex) / ex, ("%.4f (%.1f%%)" % (tl, 100 * abs(tl - ex) / ex)) if tl else "no real root"))
