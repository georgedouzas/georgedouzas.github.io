# Exercise exr-ren-x-divergent-series: partial sums of the perturbative series of <phi^2> in the zero-dimensional quartic
# theory against the exact value and the Hartree solution, lam = 0.1 and 0.5.
import mpmath as mpm
var('l'); N = 12
mom = lambda k: 0 if k % 2 else prod(range(k - 1, 0, -2))
num = sum((-l / 24)^j / factorial(j) * mom(2 + 4 * j) for j in range(N + 1)); den = sum((-l / 24)^j / factorial(j) * mom(4 * j) for j in range(N + 1))
co = [taylor(num / den, l, 0, N).coefficient(l, j) for j in range(N + 1)]
print("|a_n| / (n! (2/3)^n):", [round(float(abs(co[n]) / (factorial(n) * (2 / 3)^n)), 3) for n in range(1, N + 1)])
for lv in [0.1, 0.5]:
    f = lambda p: mpm.exp(-p**2 / 2 - lv * p**4 / 24)
    ex = float(mpm.quad(lambda p: p**2 * f(p), [-mpm.inf, 0, mpm.inf]) / mpm.quad(f, [-mpm.inf, 0, mpm.inf]))
    errs = [abs(float(sum(co[k] * lv^k for k in range(n + 1))) - ex) for n in range(N + 1)]
    best = min(range(N + 1), key=lambda n: errs[n])
    print("lam = %.1f: exact %.6f, best order %d with error %.2e, Hartree error %.2e" % (lv, ex, best, errs[best], abs(float((sqrt(1 + 2 * lv) - 1) / lv) - ex)))
