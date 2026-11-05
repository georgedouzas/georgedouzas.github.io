# Figure: the harmonic oscillator. The eigenfunctions psi_n (Hermite functions), n = 0..5, are drawn at the height of
# their energies E_n = hbar omega (n + 1/2) inside the potential m omega^2 x^2/2 (units hbar = m = omega = 1); the
# vertical marks are the classical turning points. Checks: the Hermite functions with hbar, m, omega symbolic, their
# orthonormality, the ladder action, truncated ladder matrices, the number-state uncertainties and the 3D degeneracy.
# Output: figures/qm_oscillator.svg (and an interactive version with a slider over n).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm_oscillator.svg")
x, xi = var("x xi", domain="real")
hbar, m, w = var("hbar m w", domain="positive")
ell = sqrt(hbar / (m * w))
def psi(n, X=x):
    return (m * w / (pi * hbar)) ** (1 / 4) / sqrt(2 ** n * factorial(n)) * hermite(n, X / ell) * exp(-X ** 2 / (2 * ell ** 2))
def h(n):  # dimensionless Hermite function of xi = x/ell
    return pi ** (-1 / 4) / sqrt(2 ** n * factorial(n)) * hermite(n, xi) * exp(-xi ** 2 / 2)
H = lambda f: -hbar ** 2 / (2 * m) * diff(f, x, 2) + m * w ** 2 * x ** 2 / 2 * f
print("[PASS] H psi_n = hbar omega (n + 1/2) psi_n for the Hermite functions psi_n, n = 0..6, with hbar, m, omega symbolic:",
      all(bool((H(psi(n)) - hbar * w * (n + 1 / 2) * psi(n)).simplify_full() == 0) for n in range(7)))
G = matrix(SR, 6, 6, lambda i, j: integrate(h(i) * h(j), xi, -oo, oo).simplify_full())
print("[PASS] the Hermite functions h_0..h_5 are orthonormal in L^2(R):", G == identity_matrix(6))
A = lambda f: (xi * f + diff(f, xi)) / sqrt(2); Ad = lambda f: (xi * f - diff(f, xi)) / sqrt(2)
print("[PASS] a h_n = sqrt(n) h_(n-1) and a^dagger h_n = sqrt(n+1) h_(n+1) for a = (xi + d/dxi)/sqrt 2, n = 0..6:",
      bool(A(h(0)).simplify_full() == 0) and all(bool((A(h(n)) - sqrt(n) * h(n - 1)).canonicalize_radical() == 0) for n in range(1, 7))
      and all(bool((Ad(h(n)) - sqrt(n + 1) * h(n + 1)).canonicalize_radical() == 0) for n in range(7)))
N = 6
a = matrix(SR, N, N, lambda i, j: sqrt(j) if i == j - 1 else 0); ad = a.transpose()
comm = a * ad - ad * a
print("[PASS] truncated N = 6 ladder matrices: [a, a^dagger] = diag(1, 1, 1, 1, 1, -5) (= 1 except the last entry, -(N - 1)):", comm == diagonal_matrix(SR, [1] * (N - 1) + [-(N - 1)]))
Hm = ad * a + identity_matrix(N) / 2
print("[PASS] the truncated H = a^dagger a + 1/2 is diagonal with entries n + 1/2, and x = (a + a^dagger)/sqrt 2, p = i(a^dagger - a)/sqrt 2 give H = (p^2 + x^2)/2 except in the last entry:",
      Hm == diagonal_matrix(SR, [n + 1 / 2 for n in range(N)]) and ((( -(ad - a) ** 2 + (a + ad) ** 2) / 4 - Hm)[:N - 1, :N - 1]).is_zero())
ok = True
for n in range(5):
    f = psi(n)
    vx = integrate(x ** 2 * f ** 2, x, -oo, oo).simplify_full(); vp = integrate(f * (-hbar ** 2) * diff(f, x, 2), x, -oo, oo).simplify_full()
    ok = ok and bool((vx - ell ** 2 * (n + 1 / 2)).canonicalize_radical() == 0) and bool((vp - hbar ** 2 / ell ** 2 * (n + 1 / 2)).canonicalize_radical() == 0)
print("[PASS] in |n>, (Delta x)^2 = (hbar/(m omega))(n + 1/2), (Delta p)^2 = hbar m omega (n + 1/2), so Delta x Delta p = hbar (n + 1/2), n = 0..4:", ok)
print("[PASS] the parity psi_n(-x) = (-1)^n psi_n(x), n = 0..6:", all(bool((psi(n, -x) - (-1) ** n * psi(n)).simplify_full() == 0) for n in range(7)))
deg = [len([(a1, a2, a3) for a1 in range(n + 1) for a2 in range(n + 1) for a3 in range(n + 1) if a1 + a2 + a3 == n]) for n in range(10)]
print("[PASS] the 3D isotropic oscillator level hbar omega (n + 3/2) has degeneracy (n + 1)(n + 2)/2, n = 0..9:", deg == [(n + 1) * (n + 2) // 2 for n in range(10)])
# the figure (hbar = m = omega = 1)
cols = ["firebrick", "royalblue", "darkgreen", "darkorange", "purple", "teal"]
g = plot(xi ** 2 / 2, (xi, -4, 4), color="black", thickness=2)
for n, c in zip(range(6), cols):
    e = n + 1 / 2; tp = sqrt(2 * n + 1)
    g += line([(-tp, e), (tp, e)], color="gray", linestyle="--", thickness=0.7)
    g += plot(e + 0.8 * h(n), (xi, -4, 4), color=c, thickness=1.6, legend_label=r"$\psi_%d$" % n)
    g += point([(-tp, e), (tp, e)], color=c, size=18)
g.axes_labels([r"$x/\ell$", r"$E/\hbar\omega$"]); g.set_legend_options(font_size=8, loc="upper center", ncol=6)
g.matplotlib(figsize=[8, 4.6], ymin=0, ymax=7.6).savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
xs = [-7 + 14 * k / 500.0 for k in range(501)]
frames, steps = [], []
def curves(n):
    f = fast_callable(h(n), vars=[xi], domain=RDF)
    vals = [f(u) for u in xs]
    e = n + 0.5; tp = math.sqrt(2 * n + 1)
    return [e + 0.9 * v for v in vals], [e + 2.5 * v * v for v in vals], [-tp, tp], [e, e]
for n in range(13):
    w1, d1, tx, ty = curves(n)
    frames.append({"name": str(n), "data": [{"x": xs, "y": w1}, {"x": xs, "y": d1}, {"x": tx, "y": ty}], "traces": [1, 2, 3]})
    steps.append({"label": str(n), "method": "animate", "args": [[str(n)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
w1, d1, tx, ty = curves(0)
traces = [{"x": xs, "y": [u * u / 2 for u in xs], "mode": "lines", "line": {"color": "black", "width": 2}, "name": "V = x^2/2"},
          {"x": xs, "y": w1, "mode": "lines", "line": {"color": "royalblue", "width": 2}, "name": "psi_n (offset by E_n)"},
          {"x": xs, "y": d1, "mode": "lines", "line": {"color": "firebrick", "width": 1.5}, "name": "|psi_n|^2 (offset by E_n)"},
          {"x": tx, "y": ty, "mode": "lines+markers", "line": {"color": "gray", "dash": "dash"}, "name": "classical region"}]
layout = {"title": "Harmonic oscillator (hbar = m = omega = 1): choose n", "xaxis": {"title": "x / ell", "range": [-7, 7]}, "yaxis": {"title": "E / hbar omega", "range": [0, 14.5]},
          "sliders": [{"active": 0, "currentvalue": {"prefix": "n = "}, "steps": steps, "pad": {"t": 30}}], "legend": {"orientation": "h"}}
path = plotly_html("qm_oscillator", traces, layout)
html = open(path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(path, "w").write(html)
print("wrote figures/interactive/qm_oscillator.html")
