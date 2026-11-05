# Figure: the Feynman propagator i/(p0^2 - omega^2 + i eps) of one mode (omega = 1). Left: the complex p0-plane with
# the two poles at +-sqrt(omega^2 - i eps), in the fourth and second quadrants, the real-axis contour and the
# Wick-rotated contour along the imaginary axis; the first and third quadrants (shaded) contain no pole, so the real
# axis can be turned anticlockwise onto the imaginary axis. Right: the time-dependence e^{-i omega |t|}/(2 omega) of the
# real-time propagator (real and imaginary parts) and the decaying Euclidean propagator e^{-omega |tau|}/(2 omega).
# An interactive version moves eps with a slider. Output: figures/feynman_poles.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feynman_poles.svg")
w0 = 1; epsd = 0.5
Om = sqrt(CDF(w0 ** 2, -epsd))
left = polygon([(0, 0), (2.6, 0), (2.6, 2.1), (0, 2.1)], color="lightgreen", alpha=0.25)
left += polygon([(0, 0), (-2.6, 0), (-2.6, -2.1), (0, -2.1)], color="lightgreen", alpha=0.25)
left += arrow((-2.5, 0.04), (2.5, 0.04), color="royalblue", width=1.6, arrowsize=3, legend_label="real-axis contour")
left += arrow((-0.04, -2.0), (-0.04, 2.0), color="darkorange", width=1.6, arrowsize=3, linestyle="--", legend_label="Wick-rotated contour")
left += arrow((1.7, 0.25), (0.35, 1.6), color="grey", width=0.8, arrowsize=2)
left += arrow((-1.7, -0.25), (-0.35, -1.6), color="grey", width=0.8, arrowsize=2)
left += point([(Om.real(), Om.imag()), (-Om.real(), -Om.imag())], color="firebrick", size=40, legend_label=r"poles $\pm\sqrt{\omega^2-i\epsilon}$")
left += text(r"$+\omega-i\epsilon'$", (Om.real() + 0.35, Om.imag() - 0.3), fontsize=9, color="firebrick")
left += text(r"$-\omega+i\epsilon'$", (-Om.real() - 0.35, -Om.imag() + 0.3), fontsize=9, color="firebrick")
left.axes_labels([r"$\mathrm{Re}\,p_0$", r"$\mathrm{Im}\,p_0$"]); left.set_legend_options(loc="lower right", font_size=7)
left.set_aspect_ratio(1)
t = var("t", domain="real")
right = plot(cos(w0 * abs(t)) / (2 * w0), (t, -6, 6), color="royalblue", legend_label=r"$\mathrm{Re}\,e^{-i\omega|t|}/2\omega$")
right += plot(-sin(w0 * abs(t)) / (2 * w0), (t, -6, 6), color="royalblue", linestyle="--", legend_label=r"$\mathrm{Im}\,e^{-i\omega|t|}/2\omega$")
right += plot(exp(-w0 * abs(t)) / (2 * w0), (t, -6, 6), color="darkorange", thickness=2, legend_label=r"Euclidean $e^{-\omega|\tau|}/2\omega$")
right.axes_labels([r"$t,\tau$", ""]); right.set_legend_options(loc="lower right", font_size=7)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10.5, 4]); fig.subplots_adjust(wspace=0.4); fig.savefig(OUT, bbox_inches="tight")

p0, p4, w, eps, tau = var("p0 p4 w eps tau")
assume(w > 0); assume(eps > 0); assume(tau > 0)
Omega = sqrt(w ** 2 - I * eps)
f = I / ((p0 - Omega) * (p0 + Omega))
okpoles = bool(((p0 - Omega) * (p0 + Omega)).expand() == p0 ** 2 - w ** 2 + I * eps)
r1 = f.residue(p0 == Omega); r2 = f.residue(p0 == -Omega)
quad = all(CDF(sqrt(CDF(wv ** 2, -ev))).imag() < 0 and CDF(sqrt(CDF(wv ** 2, -ev))).real() > 0 for wv in [0.5, 1, 3] for ev in [0.1, 0.01, 1e-4])
print("[PASS] the poles of i/(p0^2 - omega^2 + i eps) sit at +-sqrt(omega^2 - i eps), in the fourth and second quadrants, with residues +-i/(2 sqrt(omega^2 - i eps)):",
      okpoles and bool(r1 == I / (2 * Omega)) and bool(r2 == -I / (2 * Omega)) and quad)
# closing below for t > 0 (clockwise, -2 pi i) and above for t < 0
tt = var("tt"); assume(tt > 0)
below = -2 * pi * I * (f * exp(-I * p0 * tt)).residue(p0 == Omega) / (2 * pi)
above = 2 * pi * I * (f * exp(I * p0 * tt)).residue(p0 == -Omega) / (2 * pi)
print("[PASS] int dp0/(2 pi) i e^{-i p0 t}/(p0^2 - omega^2 + i eps) = e^{-i Omega |t|}/(2 Omega) for t > 0 and t < 0, -> e^{-i omega |t|}/(2 omega) as eps -> 0:",
      bool(below == exp(-I * Omega * tt) / (2 * Omega)) and bool(above == exp(-I * Omega * tt) / (2 * Omega)) and bool(limit(below, eps=0) == exp(-I * w * tt) / (2 * w)))
mink = 2 * pi * I * r2          # the real-axis integral of f, closing above (the integrand decays like 1/p0^2)
eucl = integrate(1 / (p4 ** 2 + w ** 2), p4, -oo, oo)
print("[PASS] Wick rotation: int dp0 i/(p0^2 - omega^2 + i eps) = pi/sqrt(omega^2 - i eps) -> pi/omega = int dp4 1/(p4^2 + omega^2):",
      bool(mink == pi / Omega) and bool(limit(mink, eps=0) == eucl) and bool(eucl == pi / w))
m, k1, k2, k3 = var("m k1 k2 k3")
DM = I / (p0 ** 2 - k1 ** 2 - k2 ** 2 - k3 ** 2 - m ** 2)
print("[PASS] substituting p0 = i p4 (dp0 = i dp4) turns i/(p^2 - m^2) dp0 into 1/(p_E^2 + m^2) dp4:",
      bool((I * DM.subs(p0=I * p4) - 1 / (p4 ** 2 + k1 ** 2 + k2 ** 2 + k3 ** 2 + m ** 2)).simplify_full() == 0))
GE = integrate(cos(p4 * tau) / (p4 ** 2 + w ** 2), p4, -oo, oo) / (2 * pi)
G = exp(-w * abs(tau)) / (2 * w)
okG = bool(GE == exp(-w * tau) / (2 * w)) and bool((-diff(exp(-w * tau), tau, 2) + w ** 2 * exp(-w * tau)) == 0) \
      and bool(diff(exp(-w * tau) / (2 * w), tau).subs(tau=0) - diff(exp(w * tau) / (2 * w), tau).subs(tau=0) == -1)
print("[PASS] the Euclidean propagator int dp/(2 pi) e^{i p tau}/(p^2 + omega^2) = e^{-omega |tau|}/(2 omega) solves (-d^2/dtau^2 + omega^2) G = delta(tau) (slope jump -1):", okG)
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
import cmath as _c
epsl = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
xs = [-3 + 6 * i / 600 for i in range(601)]
_traces = []
for ev in epsl:
    vis = abs(ev - 0.3) < 1e-9
    P = _c.sqrt(complex(1, -ev))
    _traces.append({"x": [-2.5, 2.5], "y": [0, 0], "mode": "lines", "name": "real-axis contour", "line": {"color": "royalblue", "width": 3}, "visible": vis})
    _traces.append({"x": [0, 0], "y": [-2, 2], "mode": "lines", "name": "Wick-rotated contour", "line": {"color": "darkorange", "width": 3, "dash": "dash"}, "visible": vis})
    _traces.append({"x": [P.real, -P.real], "y": [P.imag, -P.imag], "mode": "markers", "name": "poles", "marker": {"color": "firebrick", "size": 11}, "visible": vis})
    vals = [1 / complex(x0 ** 2 - 1, ev) for x0 in xs]
    _traces.append({"x": xs, "y": [v.real for v in vals], "mode": "lines", "name": "Re 1/(p0^2-1+i eps)", "line": {"color": "royalblue"}, "xaxis": "x2", "yaxis": "y2", "visible": vis})
    _traces.append({"x": xs, "y": [v.imag for v in vals], "mode": "lines", "name": "Im 1/(p0^2-1+i eps)", "line": {"color": "firebrick"}, "xaxis": "x2", "yaxis": "y2", "visible": vis})
steps = [{"method": "update", "label": "%g" % ev, "args": [{"visible": [abs(ev - ew) < 1e-12 for ew in epsl for _ in range(5)]}]} for ev in epsl]
plotly_html("feynman_poles", _traces, {"title": "Poles of the Feynman propagator (left, p0-plane) and the integrand on the real axis (right): move eps",
            "grid": {"rows": 1, "columns": 2, "pattern": "independent"}, "xaxis": {"title": "Re p0", "range": [-2.6, 2.6]}, "yaxis": {"title": "Im p0", "range": [-2.1, 2.1], "scaleanchor": "x"},
            "xaxis2": {"title": "p0 (real)"}, "yaxis2": {"range": [-12, 12]}, "legend": {"orientation": "h"},
            "sliders": [{"active": 2, "currentvalue": {"prefix": "eps = "}, "pad": {"t": 40}, "steps": steps}]})
print("wrote figures/interactive/feynman_poles.html")
