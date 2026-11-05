# Figure: a coherent state |alpha> of the oscillator (alpha = 2, units hbar = m = omega = 1, x in units of
# ell = sqrt(hbar/(m omega))). Left: the density |psi(x, t)|^2 at t = 0, pi/4, pi/2, 3 pi/4, pi, drawn at the height
# of the mean energy (|alpha|^2 + 1/2) hbar omega inside the potential: the Gaussian keeps its shape and swings like
# the classical particle. Right: <x>(t) = sqrt 2 |alpha| cos(omega t) with the band <x> +- Delta x, Delta x = 1/sqrt 2.
# Output: figures/qm_coherent_state.svg (and an interactive time slider).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm_coherent_state.svg")
xi, t = var("xi t", domain="real")
al = var("al")
def h(n):
    return pi ** (-1 / 4) / sqrt(2 ** n * factorial(n)) * hermite(n, xi) * exp(-xi ** 2 / 2)
phi = pi ** (-1 / 4) * exp(-xi ** 2 / 2 + sqrt(2) * al * xi - al ** 2 / 2)          # e^{|al|^2/2} times the normalised state
print("[PASS] a psi_alpha = alpha psi_alpha for a = (xi + d/dxi)/sqrt 2 and psi_alpha = pi^(-1/4) exp(-xi^2/2 + sqrt 2 alpha xi - alpha^2/2):",
      bool(((phi * xi + diff(phi, xi)) / sqrt(2) - al * phi).simplify_full() == 0))
Psi = exp(-I * t / 2) * phi.subs(al=al * exp(-I * t))
print("[PASS] e^(-i t/2) psi_(alpha e^(-i t)) solves i d/dt psi = (-(1/2) d^2/dxi^2 + xi^2/2) psi (the coherent state stays coherent):",
      bool((I * diff(Psi, t) - (-diff(Psi, xi, 2) / 2 + xi ** 2 / 2 * Psi)).simplify_full() == 0))
A0 = 3 / 2
NT = 40
cs = [exp(-A0 ** 2 / 2) * A0 ** n / sqrt(factorial(n)) for n in range(NT)]
ok = all(abs(RR((sum(cs[n] * h(n) for n in range(NT)) - exp(-A0 ** 2 / 2) * phi.subs(al=A0)).subs(xi=x0))) < 1e-12 for x0 in [-1, 0, 1 / 2, 2, 3])
print("[PASS] the number-state series e^(-|alpha|^2/2) sum alpha^n/sqrt(n!) |n> (40 terms, alpha = 3/2) equals the Gaussian wave function at five points:", ok)
a = matrix(RDF, NT, NT, lambda i, j: sqrt(j) if i == j - 1 else 0)
v = vector(RDF, [RDF(c) for c in cs])
print("[PASS] for the truncated 40 x 40 ladder matrix, ||a v - alpha v|| < 1e-15 and ||v|| = 1 to 1e-15 (alpha = 3/2):", (a * v - A0 * v).norm() < 1e-15 and abs(v.norm() - 1) < 1e-15)
P = [RR(c) ** 2 for c in cs]
mean = sum(n * P[n] for n in range(NT)); var_ = sum(n ** 2 * P[n] for n in range(NT)) - mean ** 2
print("[PASS] the photon-number distribution |c_n|^2 = e^(-|alpha|^2) |alpha|^(2n)/n! is Poisson with mean and variance |alpha|^2 = 9/4:", abs(mean - 9 / 4) < 1e-12 and abs(var_ - 9 / 4) < 1e-12)
# <xi>(t) for the Gaussian: the density is pi^(-1/2) exp(-(xi - sqrt 2 Re(alpha e^(-it)))^2)
dens = (exp(-4) * Psi * Psi.conjugate()).subs(al=2).simplify_full()   # e^{-|alpha|^2} normalises
print("[PASS] for alpha = 2 the density is pi^(-1/2) exp(-(xi - 2 sqrt 2 cos t)^2): <x>(t) = sqrt 2 |alpha| cos(omega t) ell, Delta x = ell/sqrt 2:",
      bool((dens - exp(-(xi - 2 * sqrt(2) * cos(t)) ** 2) / sqrt(pi)).exponentialize().simplify_full() == 0))
b = var("b")
ov = integrate(phi.subs(al=b) * phi.subs(al=al), xi, -oo, oo).simplify_full()   # real alpha, beta: <beta|alpha> e^{(a^2+b^2)/2}
print("[PASS] for real alpha, beta: <beta|alpha> = exp(-(alpha - beta)^2/2), so |<beta|alpha>|^2 = exp(-|alpha - beta|^2):",
      bool((ov * exp(-(al ** 2 + b ** 2) / 2) - exp(-(al - b) ** 2 / 2)).simplify_full() == 0))
# the figure
cols = ["firebrick", "darkorange", "darkgreen", "royalblue", "purple"]
E = 4 + 1 / 2
g1 = plot(xi ** 2 / 2, (xi, -4.5, 4.5), color="black", thickness=2) + line([(-3, E), (3, E)], color="gray", linestyle="--")
for k, c in enumerate(cols):
    tt = k * pi / 4
    g1 += plot(E + 3 * exp(-(xi - 2 * sqrt(2) * cos(tt)) ** 2) / sqrt(pi), (xi, -4.5, 4.5), color=c, thickness=1.6, legend_label=r"$\omega t = %s$" % latex(tt))
g1.axes_labels([r"$x/\ell$", r"$E/\hbar\omega$"]); g1.set_legend_options(font_size=8, loc="upper center", ncol=2); g1.set_axes_range(-4.5, 4.5, 0, 10)
tt = var("tt")
g2 = plot(2 * sqrt(2) * cos(tt), (tt, 0, 4 * pi), color="royalblue", thickness=2, legend_label=r"$\langle x\rangle/\ell$")
g2 += plot(2 * sqrt(2) * cos(tt) + 1 / sqrt(2), (tt, 0, 4 * pi), color="royalblue", linestyle="--") + plot(2 * sqrt(2) * cos(tt) - 1 / sqrt(2), (tt, 0, 4 * pi), color="royalblue", linestyle="--", legend_label=r"$\langle x\rangle \pm \Delta x$")
g2.axes_labels([r"$\omega t$", ""]); g2.set_legend_options(font_size=8)
G = graphics_array([g1, g2])
G.matplotlib(figsize=[10, 4]).savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
xs = [-5 + 10 * k / 400.0 for k in range(401)]
def fr(tt):
    q = 2 * math.sqrt(2) * math.cos(tt); pp = -2 * math.sqrt(2) * math.sin(tt)
    d = [4.5 + 3 * math.exp(-(u - q) ** 2) / math.sqrt(math.pi) for u in xs]
    r = [4.5 + 1.5 * math.pi ** (-0.25) * math.exp(-(u - q) ** 2 / 2) * math.cos(pp * u - tt / 2 - q * pp / 2) for u in xs]
    return d, r, [q], [4.5 + 0.0]
frames, steps = [], []
for k in range(0, 65):
    tt = 2 * math.pi * k / 64; nm = "%.2f" % tt
    d, r, qx, qy = fr(tt)
    frames.append({"name": nm, "data": [{"x": xs, "y": d}, {"x": xs, "y": r}, {"x": qx, "y": qy}], "traces": [1, 2, 3]})
    steps.append({"label": nm, "method": "animate", "args": [[nm], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
d, r, qx, qy = fr(0)
traces = [{"x": xs, "y": [u * u / 2 for u in xs], "mode": "lines", "line": {"color": "black", "width": 2}, "name": "V"},
          {"x": xs, "y": d, "mode": "lines", "line": {"color": "royalblue", "width": 2.5}, "name": "|psi|^2"},
          {"x": xs, "y": r, "mode": "lines", "line": {"color": "firebrick", "width": 1}, "name": "Re psi"},
          {"x": qx, "y": qy, "mode": "markers", "marker": {"size": 12, "color": "black"}, "name": "classical particle"}]
layout = {"title": "Coherent state alpha = 2 (hbar = m = omega = 1): slide omega t", "xaxis": {"title": "x / ell", "range": [-5, 5]}, "yaxis": {"title": "E / hbar omega", "range": [0, 8]},
          "sliders": [{"active": 0, "currentvalue": {"prefix": "omega t = "}, "steps": steps, "pad": {"t": 30}}], "legend": {"orientation": "h"}}
path = plotly_html("qm_coherent_state", traces, layout)
html = open(path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(path, "w").write(html)
print("wrote figures/interactive/qm_coherent_state.html")
