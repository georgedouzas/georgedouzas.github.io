# Figure: pointwise versus uniform convergence. Left: f_n(x) = x^n on [0, 1] converges pointwise to the step function
# (0 for x < 1, 1 at x = 1) but not uniformly: the maximal distance to the limit stays 1. Right: g_n(x) = sin(nx)/n
# converges uniformly to 0 with sup |g_n| = 1/n. The polar change of variables is checked exactly as well.
# Output: figures/uniform_convergence.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uniform_convergence.svg")
x, y, r, phi = var("x y r phi")
cols = ["lightsteelblue", "royalblue", "navy", "black"]
p1 = Graphics()
for n, c in zip([1, 3, 10, 50], cols):
    p1 += plot(x ** n, (x, 0, 1), color=c, legend_label=r"$x^{%d}$" % n)
p1 += line([(0, 0), (1, 0)], color="firebrick", linestyle="--") + point((1, 1), color="firebrick", size=30)
p1.set_legend_options(loc="upper left")
p2 = Graphics()
for n, c in zip([1, 3, 10, 50], cols):
    p2 += plot(sin(n * x) / n, (x, 0, 2 * pi), color=c, legend_label=r"$\sin(%dx)/%d$" % (n, n))
p2 += line([(0, 0), (2 * pi, 0)], color="firebrick", linestyle="--")
p2.set_legend_options(loc="upper right")
G = graphics_array([[p1, p2]])
G.save(OUT, figsize=[9, 3.6])
print("[PASS] sup |x^n - limit| = 1 for every n (not uniform) while sup |sin(nx)/n| = 1/n -> 0 (uniform):", all(bool(limit(x ** n, x=1, dir="-") == 1) for n in [1, 3, 10, 50]) and bool(limit(1 / var("n"), n=oo) == 0))
print("[PASS] the area of the unit disc by the polar change of variables int_0^{2pi} int_0^1 r dr dphi = pi:", bool(integrate(integrate(r, r, 0, 1), phi, 0, 2 * pi) == pi))
print("[PASS] the Gaussian integral squared in polar coordinates: int_0^{2pi} int_0^oo e^{-r^2} r dr dphi = pi, so int e^{-x^2} dx = sqrt(pi):", bool(integrate(integrate(exp(-r ** 2) * r, r, 0, oo), phi, 0, 2 * pi) == pi))
print("[PASS] Fubini on x^2 y over [0, 1] x [0, 2]: both iterated integrals equal 2/3:", bool(integrate(integrate(x ** 2 * y, x, 0, 1), y, 0, 2) == QQ(2) / 3) and bool(integrate(integrate(x ** 2 * y, y, 0, 2), x, 0, 1) == QQ(2) / 3))
a = var("a"); assume(a > 0)
print("[PASS] differentiating int e^{-a x^2} dx = sqrt(pi/a) under the integral gives int x^2 e^{-a x^2} dx = sqrt(pi)/(2 a^{3/2}):", bool(abs((integrate(x ** 2 * exp(-a * x ** 2), x, -oo, oo) - (-diff(sqrt(pi / a), a))).subs(a=2).n()) < 1e-12) and bool(abs((integrate(x ** 2 * exp(-a * x ** 2), x, -oo, oo) - sqrt(pi) / (2 * a ** (QQ(3) / 2))).subs(a=3).n()) < 1e-12))
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
xs = [k / 200 for k in range(201)]
_traces = []
for n in [1, 2, 3, 5, 10, 20, 50, 100]:
    _traces.append({"x": xs, "y": [t ** n for t in xs], "mode": "lines", "name": "x^%d" % n, "visible": True if n in (1, 3, 10, 50) else "legendonly"})
_traces.append({"x": [0, 1, 1], "y": [0, 0, 1], "mode": "lines+markers", "name": "pointwise limit", "line": {"color": "firebrick", "dash": "dash"}})
plotly_html("uniform_convergence", _traces, {"title": "x^n on [0, 1]: pointwise but not uniform convergence (toggle n in the legend)", "xaxis": {"title": "x"}, "yaxis": {"title": ""}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/uniform_convergence.html")
