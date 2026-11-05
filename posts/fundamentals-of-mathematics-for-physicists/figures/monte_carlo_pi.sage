# Figure: Monte Carlo estimation of pi. Left: 2000 seeded uniform points in the unit square, those inside the
# quarter disc x^2 + y^2 <= 1 in blue; the fraction inside estimates pi/4. Right: the error |pi_N - pi| of the
# estimate from N points for three seeds (log-log) against the predicted standard deviation 4 sqrt(p(1 - p)/N) with
# p = pi/4, which falls like 1/sqrt N. Exact checks: the variance of the indicator, the unbiasedness of the sample
# mean and of the sample variance (symbolic expectations), the maximum likelihood estimators of the Gaussian and
# Poisson families, the 95 percent Gaussian quantile and a chi-squared test on a die.
# Output: figures/monte_carlo_pi.svg.
import os, random
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monte_carlo_pi.svg")
set_random_seed(7); random.seed(int(7))
N0 = 2000
pts = [(random.random(), random.random()) for _ in range(N0)]
inside = [p for p in pts if p[0] ** 2 + p[1] ** 2 <= 1]
outside = [p for p in pts if p[0] ** 2 + p[1] ** 2 > 1]
x = var("x")
left = point(inside, color="royalblue", size=4) + point(outside, color="darkorange", size=4)
left += plot(sqrt(1 - x ** 2), (x, 0, 1), color="black", thickness=1.2)
left += line([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], color="black", thickness=0.8)
est0 = 4 * QQ(len(inside)) / N0
left += text(r"$%d$ of $%d$ inside: $\pi \approx 4 \cdot %d/%d = %.4f$" % (len(inside), N0, len(inside), N0, float(est0)), (0.5, 1.07), color="black", fontsize=9)
left.axes(False)
left.set_aspect_ratio(1)

Ns = [2 ** k for k in range(4, 18)]
right = Graphics()
cols = ["royalblue", "darkgreen", "purple"]
errs_all = []
for seed, c in zip([11, 12, 13], cols):
    random.seed(int(seed))
    count, done, errs = 0, 0, []
    for Nn in Ns:
        while done < Nn:
            u, v = random.random(), random.random()
            count += (u * u + v * v <= 1); done += 1
        errs.append(abs(4.0 * count / Nn - float(pi)))
    errs_all.append(errs)
    right += list_plot(list(zip(Ns, [max(e, 1e-5) for e in errs])), plotjoined=True, color=c, thickness=1, scale="loglog", legend_label="seed %d" % seed)
    right += list_plot(list(zip(Ns, [max(e, 1e-5) for e in errs])), color=c, size=15, scale="loglog")
p0 = float(pi) / 4
sd_pred = lambda Nn: 4 * sqrt(p0 * (1 - p0) / Nn)
right += plot(sd_pred(x), (x, Ns[0], Ns[-1]), color="firebrick", thickness=2, scale="loglog", legend_label=r"$4\sqrt{p(1-p)/N}$, $p = \pi/4$")
right.axes_labels(["$N$", r"$|\pi_N - \pi|$"])
right.set_legend_options(loc="lower left", font_size=8)
right.set_aspect_ratio("automatic")
arr = graphics_array([[left, right]])
arr.matplotlib(figsize=[10, 4.3]).savefig(OUT, bbox_inches="tight")

# exact checks
p = pi / 4
print("[PASS] the indicator of the quarter disc has mean pi/4 and variance (pi/4)(1 - pi/4), so the estimate 4 I_N has standard deviation 4 sqrt(p(1 - p)/N) = %.4f/sqrt N:" % float(4 * sqrt(p * (1 - p))), bool(p * (1 - p) == p - p ** 2) and abs(float(4 * sqrt(p * (1 - p))) - 1.6422) < 1e-3)
within = sum(1 for errs in errs_all for Nn, e in zip(Ns, errs) if e <= 3 * sd_pred(Nn))
print("[PASS] the %d seeded errors are all below three predicted standard deviations (%d of %d):" % (len(Ns) * 3, within, len(Ns) * 3), within == len(Ns) * 3)
# unbiasedness by symbolic expectation: E[x_i] = mu, E[x_i^2] = sigma^2 + mu^2, E[x_i x_j] = mu^2
mu, sigma = var("mu sigma")
def expectation(poly, xs):
    R = poly.parent()
    tot = 0
    for mono, coeff in poly.dict().items():
        term = coeff
        for i, e in enumerate(mono):
            if e == 0:
                continue
            term *= {1: mu, 2: sigma ** 2 + mu ** 2}[e]
        tot += term
    return SR(tot).expand()
ok = True
for nn in [3, 4, 5]:
    R = PolynomialRing(SR, ["x%d" % i for i in range(nn)])
    xs = R.gens()
    xbar = sum(xs) / nn
    s2 = sum((xi - xbar) ** 2 for xi in xs) / (nn - 1)
    s2_biased = sum((xi - xbar) ** 2 for xi in xs) / nn
    ok = ok and bool(expectation(xbar, xs) == mu) and bool((expectation(s2, xs) - sigma ** 2).simplify_full() == 0) and bool((expectation(s2_biased, xs) - (nn - 1) * sigma ** 2 / nn).simplify_full() == 0)
print("[PASS] estimators: E[sample mean] = mu, E[s^2] = sigma^2 with the divisor n - 1 and (n - 1)sigma^2/n with the divisor n, for n = 3, 4, 5 by symbolic expectation:", ok)
m, v = var("m v")
assume(v > 0)
data = [QQ(1), QQ(2), QQ(2), QQ(5)]
ll = sum(-(d - m) ** 2 / (2 * v) - log(2 * pi * v) / 2 for d in data)
sol = solve([diff(ll, m) == 0, diff(ll, v) == 0], [m, v], solution_dict=True)
xbar = sum(data) / len(data); vb = sum((d - xbar) ** 2 for d in data) / len(data)
print("[PASS] Gaussian maximum likelihood on the data (1, 2, 2, 5): mu_hat = sample mean 5/2 and sigma_hat^2 = biased sample variance 9/4:", any(bool(s[m] == xbar) and bool(s[v] == vb) for s in sol) and xbar == QQ(5) / 2 and vb == QQ(9) / 4)
lam = var("lam"); assume(lam > 0)
counts = [2, 0, 3, 1, 4]
llp = sum(c * log(lam) - lam - log(factorial(c)) for c in counts)
solp = solve(diff(llp, lam) == 0, lam)
print("[PASS] Poisson maximum likelihood on the counts (2, 0, 3, 1, 4): lambda_hat = sample mean 2:", bool(solp[0].rhs() == 2) and bool(diff(llp, lam, 2).subs(lam=2) < 0))
z = 1.959963984540054
print("[PASS] the 95 percent Gaussian quantile: P(|Z| <= 1.96) = erf(1.96/sqrt 2) = 0.9500 to four decimals, and P(|Z| <= 1) = 0.6827:", abs(float(erf(1.96 / sqrt(2))) - 0.95) < 1e-4 and abs(float(erf(1 / sqrt(2))) - 0.6827) < 1e-4)
obs = [8, 12, 9, 11, 7, 13]
chi2 = sum((QQ(o) - 10) ** 2 / 10 for o in obs)
C5 = RealDistribution("chisquared", 5)
pval = 1 - C5.cum_distribution_function(float(chi2))
print("[PASS] chi-squared test of a die, counts (8, 12, 9, 11, 7, 13) in 60 rolls: chi^2 = 14/5 = 2.8 with 5 degrees of freedom, p-value %.3f > 0.05, the die is not rejected:" % pval, chi2 == QQ(14) / 5 and pval > 0.05)
print("wrote", os.path.relpath(OUT))
