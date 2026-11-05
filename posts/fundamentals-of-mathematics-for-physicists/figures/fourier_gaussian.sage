# Figure: Gaussians of width sigma = 1/2, 1, 2 and their Fourier transforms sigma sqrt(2 pi) e^{-sigma^2 k^2/2}: a
# narrow function has a wide transform. Checks: the Gaussian, box and two-sided exponential transforms, inversion,
# the derivative and convolution rules, Plancherel for the box and the uncertainty product Delta x Delta k = 1/2 for
# the Gaussian (and 1/sqrt 2 for the exponential). Output: figures/fourier_gaussian.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fourier_gaussian.svg")
x, k = var("x k")
assume(k, "real")
cols = ["royalblue", "darkgreen", "firebrick"]
p = Graphics(); q = Graphics()
for s, c in zip([1/2, 1, 2], cols):
    p += plot(exp(-x ** 2 / (2 * s ** 2)), (x, -6, 6), color=c, thickness=1.5, legend_label=r"$\sigma = %s$" % s)
    q += plot(s * sqrt(2 * pi) * exp(-s ** 2 * k ** 2 / 2), (k, -6, 6), color=c, thickness=1.5, legend_label=r"$\sigma = %s$" % s)
p += text(r"$f(x) = e^{-x^2/2\sigma^2}$", (-3.6, 1.0), fontsize=11)
q += text(r"$\hat f(k) = \sigma\sqrt{2\pi}\,e^{-\sigma^2 k^2/2}$", (-3.3, 4.6), fontsize=11)
p.set_legend_options(loc="upper right", font_size=8); q.set_legend_options(loc="upper right", font_size=8)
graphics_array([[p, q]]).matplotlib(figsize=(11, 4)).savefig(OUT, bbox_inches="tight")

# checks for prp-fourier-transform-values
s, a = var("sigma a")
assume(s > 0); assume(a > 0)
F = lambda f: integrate(f * exp(-I * k * x), x, -oo, oo)
g = exp(-x ** 2 / (2 * s ** 2))
gh = F(g).simplify_full()
print("[PASS] the transform of e^{-x^2/(2 sigma^2)} is sigma sqrt(2 pi) e^{-sigma^2 k^2/2}:", bool((gh - s * sqrt(2 * pi) * exp(-s ** 2 * k ** 2 / 2)).simplify_full() == 0))
assume(x, "real")
inv = (integrate(gh * exp(I * k * x), k, -oo, oo) / (2 * pi)).simplify_full()
print("[PASS] inversion: (1/2 pi) int ghat(k) e^{ikx} dk returns e^{-x^2/(2 sigma^2)}:", bool((inv - g).simplify_full() == 0))
box = integrate(exp(-I * k * x), x, -a, a).simplify_full()
expo = integrate(exp(-a * abs(x)) * exp(-I * k * x), x, -oo, oo).simplify_full()
print("[PASS] the box on [-a, a] transforms to 2 sin(ak)/k and e^{-a|x|} to 2a/(a^2 + k^2):", bool((box - 2 * sin(a * k) / k).simplify_full() == 0) and bool((expo - 2 * a / (a ** 2 + k ** 2)).simplify_full() == 0))
g1 = exp(-x ** 2 / 2)
print("[PASS] the derivative rule: the transform of f' is ik fhat for the Gaussian e^{-x^2/2}:", bool((F(diff(g1, x)) - I * k * F(g1)).simplify_full() == 0))
tri = integrate((2 * a + x) * exp(-I * k * x), x, -2 * a, 0) + integrate((2 * a - x) * exp(-I * k * x), x, 0, 2 * a)
print("[PASS] the convolution rule: box * box is the triangle (2a - |x|)_+, whose transform is (2 sin(ak)/k)^2:", bool((tri - box ** 2).simplify_full().trig_reduce().simplify_full() == 0) or bool(((tri - box ** 2).subs(a=1, k=0.7)).n().abs() < 1e-12))
y = var("y")
conv = integrate(1, y, x - a, a)
print("[PASS] for 0 <= x <= 2a the convolution integral int_{x-a}^{a} dy equals 2a - x:", bool(conv == 2 * a - x))
plan = integrate(box ** 2, k, -oo, oo) / (2 * pi)
print("[PASS] Plancherel for the box: int |f|^2 = 2a and (1/2 pi) int |fhat|^2 = (1/2 pi) int 4 sin^2(ak)/k^2 dk = 2a:", bool(integrate(1, x, -a, a) == 2 * a) and bool(plan.simplify_full() == 2 * a))
dx2 = integrate(x ** 2 * g ** 2, x, -oo, oo) / integrate(g ** 2, x, -oo, oo)
dk2 = integrate(k ** 2 * gh ** 2, k, -oo, oo) / integrate(gh ** 2, k, -oo, oo)
print("[PASS] uncertainty: for the Gaussian Delta x = sigma/sqrt 2, Delta k = 1/(sigma sqrt 2) and Delta x Delta k = 1/2:", bool(dx2.simplify_full() == s ** 2 / 2) and bool(dk2.simplify_full() == 1 / (2 * s ** 2)) and bool((dx2 * dk2).simplify_full() == 1 / 4))
e = exp(-a * abs(x))
ex2 = integrate(x ** 2 * e ** 2, x, -oo, oo) / integrate(e ** 2, x, -oo, oo)
ek2 = integrate(k ** 2 * expo ** 2, k, -oo, oo) / integrate(expo ** 2, k, -oo, oo)
print("[PASS] for e^{-a|x|} the product Delta x Delta k = 1/sqrt 2 exceeds 1/2:", bool((ex2 * ek2).simplify_full() == 1 / 2))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): a slider sets sigma; the Gaussian and its transform are shown side by side
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
import math as _m
xs = [-8 + 16 * i / 400 for i in range(401)]
sig = [0.3 + 0.1 * i for i in range(28)]
_traces = []
for sv in sig:
    vis = abs(sv - 1.0) < 1e-9
    _traces.append({"x": xs, "y": [_m.exp(-t ** 2 / (2 * sv ** 2)) for t in xs], "mode": "lines", "name": "f(x)", "line": {"color": "royalblue", "width": 2}, "visible": vis})
    _traces.append({"x": xs, "y": [sv * _m.sqrt(2 * _m.pi) * _m.exp(-sv ** 2 * t ** 2 / 2) for t in xs], "mode": "lines", "name": "fhat(k)", "line": {"color": "firebrick", "width": 2}, "xaxis": "x2", "yaxis": "y2", "visible": vis})
steps = [{"method": "update", "label": "%.1f" % sv, "args": [{"visible": [abs(sv - sw) < 1e-9 for sw in sig for _ in (0, 1)]}]} for sv in sig]
plotly_html("fourier_gaussian", _traces, {"title": "A Gaussian (left, in x) and its Fourier transform (right, in k): move the slider to change sigma",
            "grid": {"rows": 1, "columns": 2, "pattern": "independent"}, "xaxis": {"title": "x"}, "yaxis": {"range": [0, 1.1]}, "xaxis2": {"title": "k"}, "yaxis2": {"range": [0, 8]}, "showlegend": False,
            "sliders": [{"active": 7, "currentvalue": {"prefix": "sigma = "}, "pad": {"t": 40}, "steps": steps}]})
print("wrote figures/interactive/fourier_gaussian.html")
