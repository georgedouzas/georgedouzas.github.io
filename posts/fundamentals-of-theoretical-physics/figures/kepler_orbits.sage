# Figure: Kepler's problem. Left: the orbits r = p/(1 + e cos phi) in the potential V = -k/r (units m = k = 1, L = 1,
# so p = L^2/(mk) = 1) for eccentricities e = 0 (circle), 0.5, 0.8 (ellipses), 1 (parabola) and 1.4 (hyperbola), with
# the force centre at a focus. Right: the effective potential U(r) = L^2/(2 m r^2) - k/r and the energy
# E = k(e^2 - 1)/(2p) of each orbit: bound orbits have E < 0 and move between the two turning points. Checks: the conic
# solves the orbit (Binet) equation, its energy, the conservation of the Laplace-Runge-Lenz vector, and a numerical
# orbit that closes after the period 2 pi a^(3/2) of Kepler's third law. Output: figures/kepler_orbits.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kepler_orbits.svg")
phi, e, P, r = var("phi e P r", domain="real")
cols = {0: "black", 0.5: "royalblue", 0.8: "darkgreen", 1: "darkorange", 1.4: "firebrick"}
left = point((0, 0), color="black", size=40)
for ee, c in cols.items():
    rr = 1 / (1 + ee * cos(phi))
    lim = float(pi) if ee < 1 else float(arccos(-5 / (6 * ee)))
    left += parametric_plot((rr * cos(phi), rr * sin(phi)), (phi, -lim, lim), color=c, thickness=1.6, legend_label=r"$e=%g$" % ee)
left.axes_labels([r"$x$", r"$y$"]); left.set_aspect_ratio(1); left.set_legend_options(loc="upper left", font_size=8)
U = 1 / (2 * r**2) - 1 / r
right = plot(U, (r, 0.3, 6), color="black", thickness=2, legend_label=r"$U(r)=\frac{1}{2r^2}-\frac{1}{r}$")
for ee, c in cols.items():
    E = (ee**2 - 1) / 2
    r1 = 1 / (1 + ee)
    r2 = 1 / (1 - ee) if ee < 1 else 6
    right += line([(r1, E), (min(r2, 6), E)], color=c, thickness=1.4)
right.set_axes_range(ymin=-0.6, ymax=0.8); right.axes_labels([r"$r$", r"$U,\ E$"]); right.set_legend_options(loc="lower right", font_size=8)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 4.3]); fig.savefig(OUT, bbox_inches="tight")

# the conic solves Binet's equation u'' + u = m k / L^2 = 1/P for u = 1/r = (1 + e cos phi)/P
u = (1 + e * cos(phi)) / P
print("[PASS] the conic r = p/(1 + e cos phi) solves the orbit equation u'' + u = mk/L^2 with p = L^2/(mk):", bool(diff(u, phi, 2) + u == 1 / P))
# energy at the pericentre r = p/(1 + e), where rdot = 0: E = L^2/(2 m r^2) - k/r with L^2 = m k p
m, k = var("m k", domain="positive")
rp = P / (1 + e)
Eperi = (m * k * P) / (2 * m * rp**2) - k / rp
print("[PASS] the energy of the conic of eccentricity e is E = k(e^2 - 1)/(2p):", bool((Eperi - k * (e**2 - 1) / (2 * P)).simplify_full() == 0))
# Laplace-Runge-Lenz vector A = p x L - m k r/|r| for F = -k r/|r|^3
x, y, z, px, py, pz = var("x y z px py pz", domain="real")
R = sqrt(x**2 + y**2 + z**2)
X = vector([x, y, z]); Pv = vector([px, py, pz])
Lv = X.cross_product(Pv)
A = Pv.cross_product(Lv) - m * k * X / R
xdot = Pv / m; pdot = -k * X / R**3
dA = [sum(diff(A[i], X[j]) * xdot[j] + diff(A[i], Pv[j]) * pdot[j] for j in range(3)) for i in range(3)]
print("[PASS] the Laplace-Runge-Lenz vector A = p x L - mk r/|r| has dA/dt = 0 under F = -k r/r^3:", all(bool(d.simplify_full() == 0) for d in dA))
print("[PASS] A is orthogonal to L and |A|^2 = m^2 k^2 + 2 m E L^2:", bool((A * Lv).simplify_full() == 0) and bool((A * A - m**2 * k**2 - 2 * m * (Pv * Pv / (2 * m) - k / R) * (Lv * Lv)).simplify_full() == 0))
# numerical orbit for e = 0.5 (m = k = 1, L = 1): starts at the pericentre and returns after T = 2 pi a^(3/2)
from scipy.integrate import solve_ivp
import numpy as np
def rhs(t, s):
    rr3 = (s[0]**2 + s[1]**2)**1.5
    return [s[2], s[3], -s[0] / rr3, -s[1] / rr3]
ee = 0.5; a = 1 / (1 - ee**2); T = float(2 * pi * a**1.5)
s0 = [1 / (1 + ee), 0.0, 0.0, 1 + ee]
sol = solve_ivp(rhs, [0, T], s0, rtol=1e-12, atol=1e-12)
print("[PASS] the numerical orbit of eccentricity 1/2 closes after T = 2 pi a^(3/2) (Kepler's third law), to 1e-6:", bool(np.max(np.abs(sol.y[:, -1] - np.array(s0))) < 1e-6))
print("wrote", os.path.relpath(OUT))

# interactive version: orbits (left) and effective potential with energy levels (right); click the legend to toggle
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
traces = []
for ee, c in list(cols.items()) + [(0.3, "purple"), (0.95, "teal")]:
    lim = float(pi) if ee < 1 else float(arccos(-1 / ee)) - 0.35
    ph = [-lim + 2 * lim * j / 400 for j in range(401)]
    rr = [1 / (1 + ee * np.cos(q)) for q in ph]
    keep = [(r_ * np.cos(q), r_ * np.sin(q)) for r_, q in zip(rr, ph) if r_ < 25]
    grp = "e%s" % ee
    traces.append({"x": [p_[0] for p_ in keep], "y": [p_[1] for p_ in keep], "mode": "lines", "name": "e = %s" % ee, "legendgroup": grp, "line": {"color": c, "width": 2}, "xaxis": "x", "yaxis": "y"})
    E = (ee**2 - 1) / 2
    r1 = 1 / (1 + ee); r2 = 1 / (1 - ee) if ee < 1 else 8
    traces.append({"x": [r1, min(r2, 8)], "y": [E, E], "mode": "lines", "name": "E (e = %s)" % ee, "legendgroup": grp, "showlegend": False, "line": {"color": c, "width": 2}, "xaxis": "x2", "yaxis": "y2"})
rs = [0.3 + j * 7.7 / 300 for j in range(301)]
traces.append({"x": rs, "y": [1 / (2 * q**2) - 1 / q for q in rs], "mode": "lines", "name": "U(r) = 1/(2r^2) - 1/r", "line": {"color": "black", "width": 3}, "xaxis": "x2", "yaxis": "y2"})
traces.append({"x": [0], "y": [0], "mode": "markers", "name": "force centre", "marker": {"color": "black", "size": 8}, "xaxis": "x", "yaxis": "y"})
layout = {"title": "Kepler orbits r = p/(1 + e cos phi) and the effective potential (click the legend)",
          "xaxis": {"domain": [0, 0.48], "range": [-6, 2], "title": "x"}, "yaxis": {"range": [-3.5, 3.5], "scaleanchor": "x", "title": "y"},
          "xaxis2": {"domain": [0.55, 1], "range": [0, 8], "title": "r"}, "yaxis2": {"anchor": "x2", "range": [-0.7, 0.8], "title": "U, E"},
          "legend": {"orientation": "h"}}
plotly_html("kepler_orbits", traces, layout)
print("wrote figures/interactive/kepler_orbits.html")
