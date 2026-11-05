# Figure: partial sums of three series. The geometric series sum 1/2^n converges to 1, sum 1/n^2 converges to pi^2/6,
# and the harmonic series sum 1/n diverges (its partial sums grow like log n). Output:
# figures/series_partial_sums.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "series_partial_sums.svg")
N = 40
geo = [(n, sum(1 / QQ(2) ** k for k in range(1, n + 1))) for n in range(1, N + 1)]
bas = [(n, sum(1 / QQ(k) ** 2 for k in range(1, n + 1))) for n in range(1, N + 1)]
har = [(n, sum(1 / QQ(k) for k in range(1, n + 1))) for n in range(1, N + 1)]
p = Graphics()
p += line([(0.5, 1), (N + 0.5, 1)], color="royalblue", linestyle="--") + line([(0.5, float(pi ** 2 / 6)), (N + 0.5, float(pi ** 2 / 6))], color="darkgreen", linestyle="--")
p += points([(n, float(v)) for n, v in geo], color="royalblue", size=20) + points([(n, float(v)) for n, v in bas], color="darkgreen", size=20) + points([(n, float(v)) for n, v in har], color="firebrick", size=20)
p += plot(log(x) + float(euler_gamma), (x, 1, N), color="firebrick", linestyle=":", thickness=1)
p += text(r"$\sum 1/2^n \to 1$", (33, 0.8), color="royalblue", fontsize=10) + text(r"$\sum 1/n^2 \to \pi^2/6$", (33, 1.85), color="darkgreen", fontsize=10)
p += text(r"$\sum 1/n$ diverges like $\log n + \gamma$", (26, 4.5), color="firebrick", fontsize=10)
p.axes_labels(["$n$", "partial sum $s_n$"])
p.save(OUT, figsize=[7, 4], ymin=0, ymax=5)
n = var("n")
try:
    sum(1 / n, n, 1, oo); harmonic_diverges = False
except ValueError as err:
    harmonic_diverges = "divergent" in str(err)
print("[PASS] sum 1/2^n = 1 and sum 1/n^2 = pi^2/6 exactly, and Sage reports the harmonic series as divergent:", bool(sum(1 / 2 ** n, n, 1, oo) == 1) and bool(sum(1 / n ** 2, n, 1, oo) == pi ** 2 / 6) and harmonic_diverges)
print("[PASS] the exponential series sum x^n/n! = e^x and the alternating series sum (-1)^(n+1)/n = log 2:", bool(sum(var("x") ** n / factorial(n), n, 0, oo) == exp(var("x"))) and bool(sum((-1) ** (n + 1) / n, n, 1, oo) == log(2)))
print("[PASS] the geometric partial sum s_N = 1 - 2^(-N) for N = 40:", geo[-1][1] == 1 - QQ(2) ** (-40))
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
_traces = [{"x": [k for k, _ in s], "y": [float(v) for _, v in s], "mode": "markers+lines", "line": {"width": 1}, "marker": {"size": 6}, "name": nm,
            "text": ["n = %d, s_n = %.6f" % (k, float(v)) for k, v in s], "hoverinfo": "text"} for s, nm in [(geo, "sum 1/2^k (limit 1)"), (bas, "sum 1/k^2 (limit pi^2/6)"), (har, "sum 1/k (diverges)")]]
plotly_html("series_partial_sums", _traces, {"title": "Partial sums of three series (hover for values)", "xaxis": {"title": "n"}, "yaxis": {"title": "s_n"}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/series_partial_sums.html")
