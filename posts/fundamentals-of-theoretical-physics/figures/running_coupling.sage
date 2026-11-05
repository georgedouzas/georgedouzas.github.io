# Figure: renormalisation in two pictures. Left: the one-loop running coupling of the quartic theory,
# lambda(mu) = lambda_0/(1 - (3 lambda_0/(16 pi^2)) log(mu/mu_0)), for lambda_0 = 2, 4, 8, which grows with the scale
# and blows up at the Landau pole log(mu/mu_0) = 16 pi^2/(3 lambda_0). Right: the spectrum of a scalar of mass m on
# a compact extra dimension, a circle of radius R: four-dimensional masses m_n = sqrt(m^2 + n^2/R^2), for R = 1 and
# R = 2 (m = 1/2); below the energy 1/R only the n = 0 mode is light. Output: figures/running_coupling.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "running_coupling.svg")
tt = var("tt")
lam = lambda l0: l0 / (1 - 3 * l0 / (16 * pi ** 2) * tt)
left = Graphics()
for l0, c in zip([2, 4, 8], ["royalblue", "darkorange", "firebrick"]):
    tstar = (16 * pi ** 2 / (3 * l0)).n()
    left += plot(lam(l0), (tt, -15, tstar - 0.02 * tstar), color=c, thickness=1.5, legend_label=r"$\lambda_0=%d$" % l0, ymax=40)
    left += line([(tstar, 0), (tstar, 40)], color=c, linestyle=":", thickness=1)
left.axes_labels([r"$\log(\mu/\mu_0)$", r"$\lambda(\mu)$"]); left.set_axes_range(ymin=0, ymax=40); left.set_legend_options(loc="upper left", font_size=8)
right = Graphics(); m0 = QQ(1) / 2
for R, c, dx in [(1, "royalblue", -0.15), (2, "firebrick", 0.15)]:
    pts = [(n + dx, sqrt(m0 ** 2 + n ** 2 / R ** 2)) for n in range(-6, 7)]
    right += point(pts, color=c, size=25, legend_label=r"$R=%d$" % R)
    right += line([(-6.5, 1 / R), (6.5, 1 / R)], color=c, linestyle="--", thickness=0.8)
right.axes_labels([r"$n$", r"$m_n=\sqrt{m^2+n^2/R^2}$"]); right.set_axes_range(ymin=0); right.set_legend_options(loc="upper center", font_size=8)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 3.8]); fig.subplots_adjust(wspace=0.5); fig.savefig(OUT, bbox_inches="tight")

l0, mu, b0 = var("l0 mu b0"); t = var("t")
L = l0 / (1 - 3 * l0 / (16 * pi ** 2) * t)
print("[PASS] lambda(t) = lambda_0/(1 - 3 lambda_0 t/(16 pi^2)) solves d lambda/d log mu = 3 lambda^2/(16 pi^2) with lambda(0) = lambda_0:",
      bool((diff(L, t) - 3 * L ** 2 / (16 * pi ** 2)).simplify_full() == 0) and bool(L.subs(t=0) == l0))
print("[PASS] the coupling has a Landau pole at log(mu/mu_0) = 16 pi^2/(3 lambda_0) and tends to 0 as mu -> 0:",
      bool(solve(1 / L == 0, t)[0].rhs() == 16 * pi ** 2 / (3 * l0)) and bool(limit(L.subs(l0=1), t=-oo) == 0))
# one-loop coefficient: s, t, u bubbles, each with symmetry factor 1/2, residue 2/eps of Gamma(2 - d/2)/(4 pi)^2,
# so the counterterm is (3/2) lambda^2 (1/(16 pi^2)) (2/eps) and the mu^eps dependence gives beta = 3 lambda^2/(16 pi^2)
lam_s, ep = var("lam_s ep")
residue_bubble = (gamma(2 - (4 - ep) / 2) / (4 * pi) ** 2).series(ep, 1).coefficient(ep, -1)
beta = 3 * QQ(1) / 2 * residue_bubble * lam_s ** 2
print("[PASS] the one-loop beta function of lambda phi^4/4! is (3 channels) x (symmetry factor 1/2) x (the 1/eps residue 2/(16 pi^2) of Gamma(2 - d/2)/(4 pi)^2) x lambda^2 = 3 lambda^2/(16 pi^2):",
      bool(beta == 3 * lam_s ** 2 / (16 * pi ** 2)))
# the compact extra dimension
x0, y, R, m, n = var("x0 y R m n"); assume(R > 0)
f = function("f")(x0)
Phi = f * exp(I * n * y / R)
KG5 = diff(Phi, x0, 2) - diff(Phi, y, 2) + m ** 2 * Phi        # (box_4 - d_y^2 + m^2) Phi, keeping one direction of box_4
KG4 = (diff(f, x0, 2) + (m ** 2 + n ** 2 / R ** 2) * f) * exp(I * n * y / R)
print("[PASS] Phi = phi_n(x) e^{i n y/R} solves the five-dimensional Klein-Gordon equation iff (box + m^2 + n^2/R^2) phi_n = 0: masses sqrt(m^2 + n^2/R^2):",
      bool((KG5 - KG4).simplify_full() == 0) and all(bool(exp(I * nv * (y + 2 * pi * R) / R) == exp(I * nv * y / R)) for nv in range(-3, 4)))
cnt = lambda E, Rv, mv: len([k for k in range(-1000, 1001) if mv ** 2 + k ** 2 / Rv ** 2 <= E ** 2])
okc = all(cnt(E, Rv, mv) == 2 * floor(Rv * sqrt(E ** 2 - mv ** 2)) + 1 for E, Rv, mv in [(3, 1, QQ(1) / 2), (3, 2, QQ(1) / 2), (QQ(7) / 2, 5, 1), (10, 3, 0), (QQ(1) / 2, 1, 0)])
print("[PASS] the number of tower modes with m_n <= E is 2 floor(R sqrt(E^2 - m^2)) + 1 (checked by enumeration), only n = 0 when E < sqrt(m^2 + 1/R^2):", okc)
l5, p0 = var("l5 p0")
eff = integrate(l5 / 24 * (p0 / sqrt(2 * pi * R)) ** 4, y, 0, 2 * pi * R)
print("[PASS] reducing (lambda_5/4!) phi^4 on the zero mode phi = phi_0/sqrt(2 pi R) gives the four-dimensional coupling lambda_4 = lambda_5/(2 pi R):",
      bool(eff == (l5 / (2 * pi * R)) / 24 * p0 ** 4))
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
Rs = [0.5, 1, 1.5, 2, 3, 4, 6]
_traces = []
for Rv in Rs:
    vis = abs(Rv - 1) < 1e-9
    ns = list(range(-12, 13))
    _traces.append({"x": ns, "y": [float(sqrt(0.25 + k ** 2 / Rv ** 2)) for k in ns], "mode": "markers", "name": "m_n", "marker": {"color": "firebrick", "size": 9}, "visible": vis})
    _traces.append({"x": [-12.5, 12.5], "y": [1 / Rv, 1 / Rv], "mode": "lines", "name": "1/R", "line": {"color": "grey", "dash": "dash"}, "visible": vis})
steps = [{"method": "update", "label": "%g" % Rv, "args": [{"visible": [abs(Rv - Rw) < 1e-12 for Rw in Rs for _ in range(2)]}]} for Rv in Rs]
plotly_html("running_coupling", _traces, {"title": "Masses m_n = sqrt(m^2 + n^2/R^2) of the modes on a circle of radius R (m = 1/2): move R",
            "xaxis": {"title": "n"}, "yaxis": {"title": "m_n", "range": [0, 8]}, "showlegend": False,
            "sliders": [{"active": 1, "currentvalue": {"prefix": "R = "}, "pad": {"t": 40}, "steps": steps}]})
print("wrote figures/interactive/running_coupling.html")
