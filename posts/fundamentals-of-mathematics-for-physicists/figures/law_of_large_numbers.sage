# Figure: the law of large numbers and the central limit theorem. Left: running averages of five independent
# sequences of fair coin flips (seeded) converge to 1/2; the dashed curves 1/2 +- 1/(2 sqrt n) are one standard
# deviation of the average, the dotted ones 1/2 +- 1/sqrt n are two. Right: the exact distribution of the
# standardised sum of n dice (S_n - 7n/2)/sqrt(35n/12) for n = 1, 2, 5, 20, computed by exact convolution, against
# the standard Gaussian density. Exact checks: Chebyshev's inequality on the sum of two dice, the variance of the
# sample mean, the convergence of exact probabilities to erf(1/sqrt 2), the Gaussian moments (2n-1)!! sigma^{2n},
# the count of Wick pairings, and the limit of the characteristic function cos(t/sqrt n)^n.
# Output: figures/law_of_large_numbers.svg, interactive figures/interactive/law_of_large_numbers.html.
import os, random
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "law_of_large_numbers.svg")
x, t, n, sigma = var("x t n sigma")
set_random_seed(2026); random.seed(int(2026))
N = 2000
runs = []
for r in range(5):
    flips = [random.randint(0, 1) for _ in range(N)]
    s, avg = 0, []
    for i, f in enumerate(flips, 1):
        s += f; avg.append(s / i)
    runs.append(avg)
cols = ["royalblue", "darkgreen", "darkorange", "purple", "teal"]
left = Graphics()
for avg, c in zip(runs, cols):
    left += list_plot(list(zip(range(1, N + 1), avg)), plotjoined=True, color=c, thickness=0.8, scale="semilogx")
left += plot(1 / 2 + 1 / (2 * sqrt(x)), (x, 1, N), color="black", linestyle="--", thickness=0.8, scale="semilogx")
left += plot(1 / 2 - 1 / (2 * sqrt(x)), (x, 1, N), color="black", linestyle="--", thickness=0.8, scale="semilogx")
left += plot(1 / 2 + 1 / sqrt(x), (x, 1, N), color="gray", linestyle=":", thickness=0.8, scale="semilogx")
left += plot(1 / 2 - 1 / sqrt(x), (x, 1, N), color="gray", linestyle=":", thickness=0.8, scale="semilogx")
left += line([(1, 0.5), (N, 0.5)], color="firebrick", thickness=1)
left.set_axes_range(ymin=0, ymax=1)
left.axes_labels(["$n$", "average"])
left.set_aspect_ratio("automatic")

# exact distribution of the sum of n dice by convolution
die = {i: QQ(1) / 6 for i in range(1, 7)}
def convolve(d1, d2):
    out = {}
    for a, pa in d1.items():
        for b, pb in d2.items():
            out[a + b] = out.get(a + b, 0) + pa * pb
    return out
dists = {1: die}
cur = die
for m in range(2, 31):
    cur = convolve(cur, die)
    dists[m] = cur
mean = lambda m: QQ(7) * m / 2
sd = lambda m: sqrt(QQ(35) * m / 12)
right = plot(exp(-x ** 2 / 2) / sqrt(2 * pi), (x, -4, 4), color="firebrick", thickness=2, legend_label="$N(0, 1)$")
for m, c in zip([1, 2, 5, 20], cols):
    pts = [(float((s - mean(m)) / sd(m)), float(p * sd(m))) for s, p in sorted(dists[m].items())]
    right += list_plot(pts, plotjoined=True, color=c, thickness=1.2, legend_label="$n = %d$ dice" % m)
    right += list_plot(pts, color=c, size=12)
right.axes_labels(["$z$", "probability $\\times\\,\\sigma\\sqrt{n}$"])
right.set_legend_options(loc="upper right", font_size=8)
right.set_axes_range(xmin=-4, xmax=4)
right.set_aspect_ratio("automatic")
arr = graphics_array([[left, right]])
arr.matplotlib(figsize=[10, 4.2]).savefig(OUT, bbox_inches="tight")

# exact checks
two = dists[2]; mu2, var2 = sum(s * p for s, p in two.items()), sum(s ** 2 * p for s, p in two.items()) - 49
sd2 = sqrt(var2)
cheb = all(sum(p for s, p in two.items() if abs(s - mu2) >= kk * sd2) <= 1 / kk ** 2 for kk in [1, QQ(3) / 2, 2, 3])
print("[PASS] Chebyshev for the sum of two dice (mean 7, variance 35/6): P(|S - 7| >= k sigma) <= 1/k^2 for k = 1, 3/2, 2, 3, exactly:", bool(mu2 == 7 and var2 == QQ(35) / 6) and cheb)
m = 100
pm = [(j, binomial(m, j) / QQ(2) ** m) for j in range(m + 1)]
tail = sum(p for j, p in pm if abs(QQ(j) / m - QQ(1) / 2) >= QQ(1) / 10)
print("[PASS] weak law for coin flips: the mean of n flips has variance 1/(4n); at n = 100 the exact P(|mean - 1/2| >= 1/10) = %.4f is below Chebyshev's 25/n = 1/4:" % float(tail), tail <= QQ(1) / 4 and sum((QQ(j) / m - QQ(1) / 2) ** 2 * p for j, p in pm) == QQ(1) / (4 * m))
probs = {mm: sum(p for s, p in dists[mm].items() if abs(s - mean(mm)) <= sd(mm)) for mm in [1, 2, 10, 30]}
target = float(erf(1 / sqrt(2)))
print("[PASS] central limit: P(|S_n - 7n/2| <= sigma sqrt n) for n = 1, 2, 10, 30 dice is", ["%.4f" % float(v) for v in probs.values()], "against erf(1/sqrt 2) = %.4f; the n = 30 value is within 0.02:" % target, abs(float(probs[30]) - target) < 0.02)
assume(sigma > 0)
rho = exp(-x ** 2 / (2 * sigma ** 2)) / (sqrt(2 * pi) * sigma)
moments_ok = all(bool((integrate(x ** (2 * r) * rho, x, -oo, oo) - (2 * r - 1).multifactorial(2) * sigma ** (2 * r)).simplify_full() == 0) for r in range(1, 6))
odd_ok = all(bool(integrate(x ** (2 * r + 1) * rho, x, -oo, oo) == 0) for r in range(0, 4))
print("[PASS] Gaussian moments: E X^{2n} = (2n-1)!! sigma^{2n} for n <= 5 by symbolic integration, and the odd moments vanish:", moments_ok and odd_ok)
pairings = [SetPartitions(2 * r, [2] * r).cardinality() for r in range(1, 6)]
print("[PASS] Wick pairings: the perfect matchings of 2n points number", pairings, "= (2n-1)!! for n <= 5:", pairings == [(2 * r - 1).multifactorial(2) for r in range(1, 6)])
L = limit(n * log(cos(t / sqrt(n))), n=oo)
print("[PASS] characteristic function of the standardised sum of n signs: cos(t/sqrt n)^n -> exp(-t^2/2), by the symbolic limit of n log cos(t/sqrt n) = -t^2/2:", bool(L == -t ** 2 / 2))
final = [float(r[-1]) for r in runs]
print("[PASS] the five seeded running averages at n = 2000 are", ["%.4f" % f for f in final], "all within 2/sqrt(2000) = 0.0447 of 1/2:", all(abs(f - 0.5) < 2 / sqrt(2000.0) for f in final))
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
idx = list(range(1, N + 1))
_traces = [{"x": idx, "y": [float(a) for a in avg], "mode": "lines", "name": "run %d" % (r + 1), "line": {"color": c, "width": 1.2}} for r, (avg, c) in enumerate(zip(runs, cols))]
_traces.append({"x": idx, "y": [0.5 + 0.5 / sqrt(float(i)) for i in idx], "mode": "lines", "name": "1/2 + 1/(2 sqrt n)", "line": {"color": "black", "width": 1, "dash": "dash"}})
_traces.append({"x": idx, "y": [0.5 - 0.5 / sqrt(float(i)) for i in idx], "mode": "lines", "name": "1/2 - 1/(2 sqrt n)", "line": {"color": "black", "width": 1, "dash": "dash"}})
_traces.append({"x": [1, N], "y": [0.5, 0.5], "mode": "lines", "name": "1/2", "line": {"color": "firebrick", "width": 2}})
plotly_html("law_of_large_numbers", _traces, {"title": "Running averages of fair coin flips (log n; click the legend to toggle runs)", "xaxis": {"title": "n", "type": "log"}, "yaxis": {"title": "average of the first n flips", "range": [0, 1]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/law_of_large_numbers.html")
