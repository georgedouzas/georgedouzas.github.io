# Figure: two sequences converging to the delta function. Left: the Gaussians g_eps = e^{-x^2/eps^2}/(eps sqrt pi)
# of unit integral and shrinking width; right: the Fourier kernels sin(Lx)/(pi x) = (1/2 pi) int_{-L}^{L} e^{ikx} dk
# of growing L. Checks: unit integrals, the action on test functions, the distributional derivative, theta' = delta,
# the exact action of the Fourier kernel, the principal value and the Sokhotski-Plemelj formula.
# Output: figures/delta_sequence.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delta_sequence.svg")
x = var("x")
cols = ["royalblue", "darkgreen", "darkorange", "firebrick"]
p = Graphics(); q = Graphics()
for e, c in zip([1, 1/2, 1/4, 1/8], cols):
    p += plot(exp(-x ** 2 / e ** 2) / (e * sqrt(pi)), (x, -3, 3), color=c, thickness=1.4, legend_label=r"$\varepsilon = %s$" % e)
for L, c in zip([2, 5, 15], cols):
    q += plot(sin(L * x) / (pi * x), (x, -3, 3), color=c, thickness=1.2, legend_label=r"$L = %d$" % L)
p += text(r"$g_\varepsilon(x) = e^{-x^2/\varepsilon^2}/(\varepsilon\sqrt{\pi})$", (-1.7, 5.6), fontsize=10)
q += text(r"$\delta_L(x) = \sin(Lx)/(\pi x)$", (1.7, 4.6), fontsize=10)
p.set_legend_options(loc="upper right", font_size=8); q.set_legend_options(loc="upper right", font_size=8)
graphics_array([[p, q]]).matplotlib(figsize=(11, 4)).savefig(OUT, bbox_inches="tight")

# checks for prp-distribution-identities
eps, L, k = var("epsilon L k")
assume(eps > 0); assume(L > 0); assume(k, "real")
g = exp(-x ** 2 / eps ** 2) / (eps * sqrt(pi))
print("[PASS] the Gaussians g_eps have unit integral for every eps > 0:", bool(integrate(g, x, -oo, oo) == 1))
i1 = integrate(g * cos(x), x, -oo, oo).simplify_full(); i2 = integrate(g * (x ** 2 + 1), x, -oo, oo).simplify_full()
print("[PASS] int g_eps cos x = e^{-eps^2/4} -> 1 = cos 0 and int g_eps (x^2 + 1) = 1 + eps^2/2 -> 1 as eps -> 0:", bool(i1 == exp(-eps ** 2 / 4)) and bool(i2 == 1 + eps ** 2 / 2) and bool(limit(i1, epsilon=0) == 1) and bool(limit(i2, epsilon=0) == 1))
d1 = integrate(diff(g, x) * sin(x), x, -oo, oo).simplify_full()
print("[PASS] the derivative: int g_eps'(x) sin x dx = -e^{-eps^2/4} -> -1 = -(sin)'(0), the action of delta':", bool(d1 == -exp(-eps ** 2 / 4)) and bool(limit(d1, epsilon=0) == -1))
xd = integrate(x * diff(g, x) * cos(x), x, -oo, oo).simplify_full()
print("[PASS] x delta' = -delta: int x g_eps'(x) cos x dx -> -1 = -cos 0:", bool(limit(xd, epsilon=0) == -1))
f = exp(-x ** 2) * (1 + x)
print("[PASS] theta' = delta: -int_0^oo f'(x) dx = f(0) for f = (1 + x) e^{-x^2}:", bool(-integrate(diff(f, x), x, 0, oo) == f.subs(x=0)))
dl = integrate(exp(I * k * x), k, -L, L) / (2 * pi)
print("[PASS] (1/2 pi) int_{-L}^{L} e^{ikx} dk = sin(Lx)/(pi x), of unit integral for every L > 0:", bool((dl - sin(L * x) / (pi * x)).simplify_full() == 0) and bool(integrate(sin(L * x) / (pi * x), x, -oo, oo) == 1))
forget(L > 0); assume(L > 1)
print("[PASS] int sin(Lx)/(pi x) cos x dx = 1 = cos 0 exactly for every L > 1 (the kernel acts as delta on e^{ix}):", bool(integrate(sin(L * x) * cos(x) / (pi * x), x, -oo, oo) == 1))
assume(eps < 1)
pv = limit(integrate(1 / x, x, -1, -eps) + integrate(1 / x, x, eps, 2), epsilon=0)
print("[PASS] the principal value PV int_{-1}^{2} dx/x = log 2:", bool(pv == log(2)))
print("[PASS] Sokhotski-Plemelj: Im 1/(x - i eps) = eps/(x^2 + eps^2) has integral pi for every eps > 0, and int sin(kx)/x dx = pi sgn k:", bool(integrate(eps / (x ** 2 + eps ** 2), x, -oo, oo) == pi) and bool(integrate(sin(k * x) / x, x, -oo, oo) == pi * sgn(k)))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): a slider sets eps; the Gaussian g_eps is shown with its action on cos x
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
import math as _m
xs = [-3 + 6 * i / 600 for i in range(601)]
es = [1.0 - 0.05 * i for i in range(19)]
_traces = [{"x": xs, "y": [_m.cos(t) for t in xs], "mode": "lines", "name": "test function cos x", "line": {"color": "gray", "dash": "dot"}}]
for ev in es:
    _traces.append({"x": xs, "y": [_m.exp(-t ** 2 / ev ** 2) / (ev * _m.sqrt(_m.pi)) for t in xs], "mode": "lines", "name": "g_eps", "line": {"color": "royalblue", "width": 2}, "visible": abs(ev - 0.5) < 1e-9})
steps = [{"method": "update", "label": "%.2f" % ev, "args": [{"visible": [True] + [abs(ev - ew) < 1e-9 for ew in es]}, {"title": "g_eps with eps = %.2f: int g_eps cos x dx = exp(-eps^2/4) = %.4f" % (ev, _m.exp(-ev ** 2 / 4))}]} for ev in es]
plotly_html("delta_sequence", _traces, {"title": "g_eps with eps = 0.50: int g_eps cos x dx = exp(-eps^2/4) = %.4f" % _m.exp(-0.25 / 4), "xaxis": {"title": "x"}, "yaxis": {"range": [-1.2, 6]}, "showlegend": False,
            "sliders": [{"active": 10, "currentvalue": {"prefix": "eps = "}, "pad": {"t": 40}, "steps": steps}]})
print("wrote figures/interactive/delta_sequence.html")
