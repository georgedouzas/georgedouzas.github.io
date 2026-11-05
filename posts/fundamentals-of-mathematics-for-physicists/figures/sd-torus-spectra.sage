# Figure: eigenvalues -(m + alpha)/R of the Dirac operator i d/dy on a circle of radius R = 1 with the twisted boundary
# condition psi(y + 2 pi) = exp(2 pi i alpha) psi(y), as functions of the twist alpha in [0, 1]; alpha = 0 is the periodic
# and alpha = 1/2 the antiperiodic spin structure. Output: figures/sd-torus-spectra.svg and an interactive version.
import os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-torus-spectra.svg")
a = var("a")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
al = [k / 100 for k in range(101)]
for m in range(-3, 3):
    ax.plot(al, [-(m + t) for t in al], color="royalblue", lw=1.5)
ax.axvline(0.5, color="firebrick", ls="--"); ax.text(0.52, 2.7, "antiperiodic", color="firebrick", fontsize=9)
ax.plot([0], [0], "ko"); ax.text(0.03, 0.2, "zero mode (periodic)", fontsize=9)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlim(0, 1); ax.set_ylim(-3, 3); ax.set_xlabel(r"twist $\alpha$"); ax.set_ylabel(r"eigenvalue $\lambda R$")
fig.tight_layout(); fig.savefig(OUT)
y = var("y")
print("[PASS] i d/dy e^{i (m + alpha) y} = -(m + alpha) e^{i (m + alpha) y} and the mode obeys the twisted condition:", all(bool(I * diff(exp(I * (m + a) * y), y) == -(m + a) * exp(I * (m + a) * y)) and bool((exp(I * (m + a) * (y + 2 * pi)) / exp(I * (m + a) * y)).simplify_full() == exp(2 * pi * I * (m + a)).simplify_full()) for m in (-1, 0, 2)))
print("[PASS] the smallest |eigenvalue| is min(alpha, 1 - alpha)/R, maximal 1/2 at alpha = 1/2:", all(abs(min([abs(m + al) for m in range(-3, 3)]) - min(al, 1 - al)) < 1e-12 for al in (0, 0.1, 0.25, 0.5, 0.8)))
print("[PASS] a zero mode exists only for alpha = 0:", [al for al in (0, 0.25, 0.5, 0.75) if min(abs(m + al) for m in range(-3, 3)) == 0] == [0])
print("wrote", os.path.relpath(OUT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
xs = [k / 200 for k in range(201)]
traces = []
for m in range(-4, 4):
    traces.append({"x": xs, "y": [-(m + t) for t in xs], "mode": "lines", "name": "circle, m = %d" % m, "line": {"color": "royalblue"}, "showlegend": m == 0, "legendgroup": "circle"})
# square torus T^2 (R = 1), twist alpha in the first direction, periodic in the second: eigenvalues +- |k|
import itertools
levels = sorted(set(round(float(sqrt((m1 + t) ** 2 + m2 ** 2)), 9) for t in [0] for m1, m2 in itertools.product(range(-3, 4), repeat=2)))
for m1, m2 in itertools.product(range(-2, 3), range(0, 3)):
    traces.append({"x": xs, "y": [float(sqrt((m1 + t) ** 2 + m2 ** 2)) for t in xs], "mode": "lines", "name": "T^2, |k|", "line": {"color": "firebrick", "width": 1}, "showlegend": (m1, m2) == (0, 0), "legendgroup": "torus", "visible": "legendonly"})
plotly_html("sd-torus-spectra", traces, {"title": "Dirac eigenvalues on the circle (and |k| on the square torus) against the twist alpha", "xaxis": {"title": "alpha"}, "yaxis": {"title": "lambda R", "range": [-3, 3]}})
print("wrote figures/interactive/sd-torus-spectra.html")
