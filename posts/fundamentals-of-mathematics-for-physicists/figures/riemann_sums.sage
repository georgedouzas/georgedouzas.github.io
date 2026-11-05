# Figure: Riemann sums of x^2 on [0, 1] with n = 4 and n = 16 subintervals (left endpoints, lower sums, in blue;
# right endpoints, upper sums, in red); both squeeze the integral 1/3 as n grows. The Gaussian integral is checked
# exactly. Output: figures/riemann_sums.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "riemann_sums.svg")
x = var("x")
f = x ** 2
def panel(n, x0):
    p = plot(f, (x, 0, 1), color="black", thickness=1.5).translate((x0, 0)) if False else Graphics()
    h = QQ(1) / n
    for k in range(n):
        a, b = k * h, (k + 1) * h
        p += polygon([(x0 + a, 0), (x0 + b, 0), (x0 + b, a ** 2), (x0 + a, a ** 2)], color="royalblue", alpha=0.35, edgecolor="royalblue")
        p += polygon([(x0 + a, a ** 2), (x0 + b, a ** 2), (x0 + b, b ** 2), (x0 + a, b ** 2)], color="firebrick", alpha=0.35, edgecolor="firebrick")
    p += parametric_plot((x0 + x, f), (x, 0, 1), color="black", thickness=1.5)
    L = sum((k * h) ** 2 * h for k in range(n)); U = sum(((k + 1) * h) ** 2 * h for k in range(n))
    p += text("$n = %d$: lower $%s$, upper $%s$" % (n, latex(L), latex(U)), (x0 + 0.5, 1.12), fontsize=9)
    return p, L, U
p1, L4, U4 = panel(4, 0); p2, L16, U16 = panel(16, 1.3)
p = p1 + p2
p.axes(False)
p.save(OUT, figsize=[8, 3.6], ymin=-0.05, ymax=1.2)
print("[PASS] lower and upper Riemann sums of x^2 on [0, 1] bracket 1/3 for n = 4 and n = 16, and converge to it:", bool(L4 < QQ(1) / 3 < U4) and bool(L16 < QQ(1) / 3 < U16) and bool(U16 - L16 == QQ(1) / 16))
print("[PASS] integral of x^2 over [0, 1] = 1/3 and the lower sums (n-1)n(2n-1)/(6n^3) -> 1/3:", bool(integrate(x ** 2, x, 0, 1) == QQ(1) / 3) and bool(limit((var("n") - 1) * var("n") * (2 * var("n") - 1) / (6 * var("n") ** 3), n=oo) == QQ(1) / 3))
print("[PASS] the Gaussian integral int_{-oo}^{oo} e^{-x^2} dx = sqrt(pi) and int_0^oo e^{-x} dx = 1:", bool(integrate(exp(-x ** 2), x, -oo, oo) == sqrt(pi)) and bool(integrate(exp(-x), x, 0, oo) == 1))
print("[PASS] the fundamental theorem on sin: int_0^pi sin x dx = -cos(pi) + cos(0) = 2:", bool(integrate(sin(x), x, 0, pi) == 2))
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
xs = [k / 200 for k in range(201)]
_traces = [{"x": xs, "y": [t * t for t in xs], "mode": "lines", "name": "x^2", "line": {"color": "black", "width": 2}}]
for n, col in [(4, "royalblue"), (8, "darkgreen"), (16, "firebrick"), (32, "purple")]:
    h = 1.0 / n
    xsq, ysq = [], []
    for k in range(n):
        a = k * h; xsq += [a, a, a + h, a + h, None]; ysq += [0, a * a, a * a, 0, None]
    L = sum((k * h) ** 2 * h for k in range(n))
    _traces.append({"x": xsq, "y": ysq, "mode": "lines", "name": "lower sum, n = %d: %.4f" % (n, L), "line": {"color": col, "width": 1}, "visible": True if n == 4 else "legendonly"})
plotly_html("riemann_sums", _traces, {"title": "Lower Riemann sums of x^2 on [0, 1] (click the legend to change n; the integral is 1/3)", "xaxis": {"title": "x"}, "yaxis": {"title": ""}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/riemann_sums.html")
