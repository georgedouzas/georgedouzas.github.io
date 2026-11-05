# Figure: the binomial distributions Binomial(n, 1/2) for n = 4, 16, 64 (top row) and the Poisson distributions
# Poisson(lambda) for lambda = 1, 4, 16 (bottom row), each with the Gaussian density of the same mean and variance
# (red curve): both families approach the Gaussian as n or lambda grows (de Moivre-Laplace, a case of the central
# limit theorem). Exact checks: normalisation, mean and variance of the binomial (rational arithmetic), of the
# Poisson (symbolic sums) and of the Gaussian (symbolic integrals), the characteristic functions, and the Poisson
# limit of the binomial. Output: figures/binomial_poisson_gaussian.svg and the interactive
# figures/interactive/binomial_poisson_gaussian.html (toggle n in the legend).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binomial_poisson_gaussian.svg")
x, t, k, lam, mu, sigma = var("x t k lam mu sigma")

def bars(pmf, mean, variance, title):
    g = Graphics()
    for j, pj in pmf:
        if pj > 1e-6:
            g += polygon2d([(j - 0.4, 0), (j + 0.4, 0), (j + 0.4, pj), (j - 0.4, pj)], color="lightsteelblue", edgecolor="steelblue", thickness=0.6)
    s = sqrt(float(variance))
    gauss = exp(-(x - float(mean)) ** 2 / (2 * float(variance))) / (s * sqrt(2 * pi))
    g += plot(gauss, (x, max(-0.6, float(mean) - 4 * s), float(mean) + 4 * s), color="firebrick", thickness=1.5)
    g += text(title, (float(mean) + 2.2 * s, 1.0 * max(pj for _, pj in pmf)), color="black", fontsize=9)
    g.set_aspect_ratio("automatic")
    return g

binom_pmf = lambda n, p: [(j, binomial(n, j) * p ** j * (1 - p) ** (n - j)) for j in range(n + 1)]
pois_pmf = lambda l, kmax: [(j, exp(-l) * l ** j / factorial(j)) for j in range(kmax + 1)]
row1 = [bars([(j, float(pj)) for j, pj in binom_pmf(n, QQ(1) / 2)], QQ(n) / 2, QQ(n) / 4, "Binomial(%d, 1/2)" % n) for n in [4, 16, 64]]
row2 = [bars([(j, float(pj)) for j, pj in pois_pmf(l, l + 4 * int(sqrt(l)) + 4)], l, l, "Poisson(%d)" % l) for l in [1, 4, 16]]
arr = graphics_array([row1, row2])
arr.matplotlib(figsize=[10, 5.2]).savefig(OUT, bbox_inches="tight")

# exact checks: binomial
ok = True
for n, p in [(10, QQ(1) / 3), (20, QQ(1) / 4), (64, QQ(1) / 2)]:
    pmf = binom_pmf(n, p)
    tot = sum(pj for _, pj in pmf); m1 = sum(j * pj for j, pj in pmf); m2 = sum(j ** 2 * pj for j, pj in pmf)
    ok = ok and tot == 1 and m1 == n * p and m2 - m1 ** 2 == n * p * (1 - p)
print("[PASS] Binomial(n, p): probabilities sum to 1, mean np, variance np(1 - p) for (n, p) = (10, 1/3), (20, 1/4), (64, 1/2), exactly:", ok)
n0, p0 = 5, var("p")
cf_binom = sum(binomial(n0, j) * p0 ** j * (1 - p0) ** (n0 - j) * exp(I * t * j) for j in range(n0 + 1))
print("[PASS] the binomial characteristic function is (1 - p + p e^{it})^n (n = 5, symbolic in p and t):", bool((cf_binom - (1 - p0 + p0 * exp(I * t)) ** n0).expand().simplify_full() == 0))
# Poisson
assume(lam > 0); assume(k, "integer")
pk = exp(-lam) * lam ** k / factorial(k)
S0, S1, S2 = [SR(sum(k ** r * pk, k, 0, oo)).simplify_full() for r in [0, 1, 2]]
cf_pois = SR(sum(exp(I * t * k) * pk, k, 0, oo)).simplify_full()
print("[PASS] Poisson(lambda): sum 1, mean lambda, variance lambda and characteristic function exp(lambda (e^{it} - 1)), by symbolic summation:", bool(S0 == 1) and bool(S1 == lam) and bool((S2 - S1 ** 2).simplify_full() == lam) and bool((cf_pois - exp(lam * (exp(I * t) - 1))).simplify_full() == 0))
# Gaussian
assume(sigma > 0)
rho = exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt(2 * pi) * sigma)
I0 = integrate(rho, x, -oo, oo).simplify_full(); I1 = integrate(x * rho, x, -oo, oo).simplify_full(); I2 = integrate((x - mu) ** 2 * rho, x, -oo, oo).simplify_full()
cf_gauss = integrate(exp(I * t * x) * rho, x, -oo, oo).simplify_full()
print("[PASS] N(mu, sigma^2): the density integrates to 1, has mean mu, variance sigma^2 and characteristic function exp(i mu t - sigma^2 t^2/2):", bool(I0 == 1) and bool(I1 == mu) and bool(I2 == sigma ** 2) and bool((cf_gauss - exp(I * mu * t - sigma ** 2 * t ** 2 / 2)).simplify_full() == 0))
# Poisson limit of the binomial
diffs = [max(abs(float(binomial(n, j) * (QQ(2) / n) ** j * (1 - QQ(2) / n) ** (n - j)) - float(exp(-2) * 2 ** j / factorial(j))) for j in range(6)) for n in [10, 100, 1000]]
print("[PASS] Poisson limit: Binomial(n, 2/n) probabilities at k <= 5 approach e^{-2} 2^k/k!, max difference", ["%.4f" % d for d in diffs], "decreasing and below 1/1000 at n = 1000:", diffs[0] > diffs[1] > diffs[2] and diffs[2] < 1e-3)
# de Moivre-Laplace: exact binomial probability of |X - n/2| <= sqrt(n)/2 vs erf(1/sqrt 2)
n = 400
pm = binom_pmf(n, QQ(1) / 2)
pr = sum(pj for j, pj in pm if abs(j - n / 2) <= sqrt(n) / 2)
print("[PASS] de Moivre-Laplace: for Binomial(400, 1/2), P(|X - 200| <= 10) = %.4f is within 0.03 of P(|Z| <= 1) = erf(1/sqrt 2) = %.4f:" % (float(pr), float(erf(1 / sqrt(2)))), abs(float(pr) - float(erf(1 / sqrt(2)))) < 0.03)
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
_traces = []
cols = ["royalblue", "darkgreen", "darkorange", "purple", "teal", "goldenrod", "brown"]
for n, c in zip([2, 4, 8, 16, 32, 64, 128], cols):
    pmf = binom_pmf(n, QQ(1) / 2)
    s = float(sqrt(QQ(n) / 4))
    xs = [float((j - QQ(n) / 2) / s) for j, _ in pmf]
    ys = [float(pj) * s for _, pj in pmf]
    _traces.append({"x": xs, "y": ys, "mode": "lines+markers", "name": "Binomial(%d, 1/2), standardised" % n, "line": {"color": c, "width": 1.5}, "marker": {"size": 5}, "visible": True if n in (4, 32) else "legendonly"})
zs = [-4 + 8 * i / 200 for i in range(201)]
_traces.append({"x": zs, "y": [float(exp(-z ** 2 / 2) / sqrt(2 * pi)) for z in zs], "mode": "lines", "name": "N(0, 1) density", "line": {"color": "firebrick", "width": 3}})
plotly_html("binomial_poisson_gaussian", _traces, {"title": "Standardised binomial distributions (k - n/2)/sqrt(n/4) against the Gaussian (click the legend to toggle n)", "xaxis": {"title": "standardised value", "range": [-4, 4]}, "yaxis": {"title": "probability x sqrt(n/4)"}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/binomial_poisson_gaussian.html")
