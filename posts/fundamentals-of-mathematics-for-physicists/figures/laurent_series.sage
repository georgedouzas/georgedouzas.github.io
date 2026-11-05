# Figure: Laurent series of f(z) = 1/((z - 1)(z - 2)). Left: the three regions |z| < 1, 1 < |z| < 2, |z| > 2 in which
# f has three different expansions, with the two poles. Right: the error |S_N(z) - f(z)| of the partial sums of the
# annulus expansion at z = 1.5 i, decreasing geometrically in N. Output: figures/laurent_series.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laurent_series.svg")
z = var("z")
f = 1/((z - 1)*(z - 2))
p = circle((0, 0), 2.9, fill=True, color="lightyellow", alpha=0.8, edgecolor="lightyellow", zorder=0)
p += circle((0, 0), 2, fill=True, color="lightgreen", alpha=0.5, zorder=1) + circle((0, 0), 1, fill=True, color="lightblue", alpha=0.8, zorder=2)
p += circle((0, 0), 1, color="royalblue", zorder=3) + circle((0, 0), 2, color="darkgreen", zorder=3)
p += point((1, 0), size=50, color="firebrick", marker="x", zorder=6) + point((2, 0), size=50, color="firebrick", marker="x", zorder=6)
p += text("|z| < 1: Taylor", (0, 0.45), fontsize=9, zorder=6) + text("1 < |z| < 2: Laurent", (0, 1.25), fontsize=9, zorder=6) + text("|z| > 2: Laurent", (0, 2.45), fontsize=9, zorder=6)
p += point((0, 1.5), size=35, color="black", zorder=6) + text("z = 1.5 i", (0.6, 1.65), fontsize=9, zorder=6)
p += text("poles at 1 and 2", (1.5, -0.3), fontsize=9, color="firebrick", zorder=6)
p.axes(False); p.set_aspect_ratio(1)
# the annulus expansion: 1/((z-1)(z-2)) = 1/(z-2) - 1/(z-1) = -(1/2) sum_{n>=0} (z/2)^n - sum_{n>=1} z^{-n}
def S(N, w):
    return sum(-(w/2)^n/2 for n in range(0, N + 1)) + sum(-w^(-n) for n in range(1, N + 1))
w0 = 1.5*I
errs = [(N, abs(S(N, w0) - f.subs(z=w0)).n()) for N in range(1, 25)]
q = list_plot([(N, e) for N, e in errs], scale="semilogy", color="darkgreen", plotjoined=True, marker="o")
q.axes_labels(["N", "|S_N(1.5 i) - f(1.5 i)|"])
graphics_array([[p, q]]).matplotlib(figsize=(10, 4.4)).savefig(OUT, bbox_inches="tight")
# checks for prp-residue-integrals and the series
x = var("x", domain="real")
print("[PASS] partial fractions: 1/((z - 1)(z - 2)) = 1/(z - 2) - 1/(z - 1):", bool((f - (1/(z - 2) - 1/(z - 1))).simplify_full() == 0))
print("[PASS] the annulus partial sums converge: the error at z = 1.5 i falls below 1e-3 by N = 24 and decreases from N to N + 2 at every step:", errs[-1][1] < 1e-3 and all(errs[k + 2][1] < errs[k][1] for k in range(len(errs) - 2)))
print("[PASS] residues: Res_{z=1} f = -1, Res_{z=2} f = 1, Res_{z=i} 1/(1 + z^2) = 1/(2i) = -i/2:", bool(f.residue(z == 1) == -1) and bool(f.residue(z == 2) == 1) and bool((1/(1 + z^2)).residue(z == I) == -I/2))
print("[PASS] the integral of 1/(1 + x^2) over the real line is pi = 2 pi i Res_{z=i}:", bool(integrate(1/(1 + x^2), x, -oo, oo) == pi) and bool((2*pi*I*(1/(1 + z^2)).residue(z == I)).simplify_full() == pi))
r4 = sum((1/(1 + z^4)).residue(z == w) for w in (exp(I*pi/4), exp(3*I*pi/4)))
print("[PASS] the integral of 1/(1 + x^4) over the real line is pi/sqrt(2) = 2 pi i (sum of the two upper residues):", bool(integrate(1/(1 + x^4), x, -oo, oo) == pi/sqrt(2)) and bool((2*pi*I*r4 - pi/sqrt(2)).simplify_full() == 0))
print("[PASS] the integral of cos x/(1 + x^2) over the real line is pi/e = 2 pi i Res_{z=i} e^{iz}/(1 + z^2):", bool(integrate(cos(x)/(1 + x^2), x, -oo, oo) == pi*exp(-1)) and bool((2*pi*I*(exp(I*z)/(1 + z^2)).residue(z == I)).simplify_full() == pi*exp(-1)))
print("[PASS] the Laurent series of 1/sin z at 0 starts 1/z + z/6 + 7 z^3/360 (a simple pole with residue 1):", bool((1/sin(z)).series(z == 0, 4).truncate() == 1/z + z/6 + 7*z^3/360) and bool((1/sin(z)).residue(z == 0) == 1))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the partial sums S_N along the circle |z| = 1.5 against f, for several N
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
import cmath as _cm
_th = [2*_cm.pi*k/180 for k in range(181)]
_ws = [1.5*_cm.exp(1j*t) for t in _th]
def _SN(N, w):
    return sum(-(w/2)**n/2 for n in range(0, N + 1)) + sum(-w**(-n) for n in range(1, N + 1))
_traces = [{"x": [t for t in _th], "y": [abs(1/((w - 1)*(w - 2))) for w in _ws], "mode": "lines", "line": {"color": "black", "width": 3}, "name": "|f(1.5 e^(i theta))|"}]
for _N, _c in [(1, "lightsalmon"), (2, "orange"), (4, "darkorange"), (8, "firebrick")]:
    _traces.append({"x": [t for t in _th], "y": [abs(_SN(_N, w)) for w in _ws], "mode": "lines", "line": {"color": _c, "dash": "dot"}, "name": "|S_%d|" % _N})
plotly_html("laurent_series", _traces, {"title": "Partial sums of the Laurent series on the circle |z| = 1.5 (hover for values)",
            "xaxis": {"title": "theta"}, "yaxis": {"title": "modulus", "range": [0, 2.5]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/laurent_series.html")
