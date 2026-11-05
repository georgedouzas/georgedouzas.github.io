# Figure: Taylor polynomials of sin x at 0 of degrees 1, 3, 5, 7 approach the function on a growing interval; the
# remainder of the degree-7 polynomial is bounded by |x|^9/9!. Output: figures/taylor_sin.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taylor_sin.svg")
x = var("x")
p = plot(sin(x), (x, -2 * pi, 2 * pi), color="black", thickness=2, legend_label=r"$\sin x$")
cols = ["royalblue", "darkgreen", "darkorange", "firebrick"]
for d, c in zip([1, 3, 5, 7], cols):
    T = taylor(sin(x), x, 0, d)
    p += plot(T, (x, -2 * pi, 2 * pi), color=c, thickness=1.2, legend_label=r"$T_%d$" % d)
p.set_legend_options(loc="lower right")
p.save(OUT, figsize=[7, 4], ymin=-2, ymax=2)
T7 = taylor(sin(x), x, 0, 7)
print("[PASS] T_7(x) = x - x^3/6 + x^5/120 - x^7/5040:", bool(T7 == x - x ** 3 / 6 + x ** 5 / 120 - x ** 7 / 5040))
print("[PASS] |sin x - T_7(x)| <= |x|^9/9! at x = 1, 2, 3 (Taylor's remainder):", all(bool(abs(sin(t) - T7.subs(x=t)).n() <= (t ** 9 / factorial(9)).n()) for t in [1, 2, 3]))
print("[PASS] derivatives: (sin x^2)' = 2x cos x^2, (x^2 e^x)' = (x^2 + 2x) e^x, (log x)' = 1/x:", bool(diff(sin(x ** 2), x) == 2 * x * cos(x ** 2)) and bool((diff(x ** 2 * exp(x), x) - (x ** 2 + 2 * x) * exp(x)).simplify_full() == 0) and bool(diff(log(x), x) == 1 / x))
print("[PASS] the mean value theorem on x^3 over [0, 2]: the slope 4 is attained at c = 2/sqrt(3) in (0, 2):", bool(diff(x ** 3, x).subs(x=2 / sqrt(3)) == 4) and bool(0 < 2 / sqrt(3) < 2))
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
xs = [-2 * float(pi) + k * 4 * float(pi) / 300 for k in range(301)]
_traces = [{"x": xs, "y": [float(sin(t)) for t in xs], "mode": "lines", "name": "sin x", "line": {"color": "black", "width": 3}}]
for d, c in zip([1, 3, 5, 7, 9, 11], cols + ["purple", "teal"]):
    T = taylor(sin(x), x, 0, d)
    _traces.append({"x": xs, "y": [max(-3, min(3, float(T.subs(x=t)))) for t in xs], "mode": "lines", "name": "T_%d" % d, "line": {"color": c, "width": 1.5}, "visible": True if d <= 7 else "legendonly"})
plotly_html("taylor_sin", _traces, {"title": "Taylor polynomials of sin x (click the legend to add T_9, T_11)", "xaxis": {"title": "x"}, "yaxis": {"title": "", "range": [-2, 2]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/taylor_sin.html")
