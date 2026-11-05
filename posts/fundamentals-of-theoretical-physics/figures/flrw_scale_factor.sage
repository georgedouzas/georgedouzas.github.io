# Figure: the FLRW metric and the Friedmann equations, signature (+, -, -, -), units c = 1:
# ds^2 = dt^2 - a(t)^2 (dr^2/(1 - k r^2) + r^2 dth^2 + r^2 sin^2 th dph^2).
# SageManifolds computes the Einstein tensor; with a perfect fluid T_mu nu = (rho + p) u_mu u_nu - p g_mu nu the
# Einstein equations G = 8 pi G T become the two Friedmann equations, whose consequence is the continuity equation.
# The scale factor of a flat universe filled with radiation, matter or a cosmological constant, and of the mixture
# Omega_m = 0.3, Omega_Lambda = 0.7, is plotted against H0 (t - t0), normalised to a(t0) = 1 and adot(t0) = H0.
# Output: figures/flrw_scale_factor.svg and figures/interactive/flrw_scale_factor.html.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flrw_scale_factor.svg")
import numpy as np
M4 = Manifold(4, "M", structure="Lorentzian")
X = M4.chart(r"t r:(0,+oo) th:(0,pi):\theta ph:(0,2*pi):\phi"); t, r, th, ph = X[:]
k = var("k"); a = function("a")(t)
g = M4.metric("g")
g[0, 0] = 1; g[1, 1] = -a^2/(1 - k*r^2); g[2, 2] = -a^2*r^2; g[3, 3] = -a^2*r^2*sin(th)^2
Ric = g.ricci(); Rs = g.ricci_scalar()
Gt = Ric - (Rs/2)*g
ad, add = diff(a, t), diff(a, t, 2)
ok_tt = bool((Gt[0, 0].expr() - 3*(ad^2 + k)/a^2).simplify_full() == 0)
ok_ii = all(bool((Gt[i, i].expr() + (2*a*add + ad^2 + k)*g[i, i].expr()/(-a^2)).simplify_full() == 0) for i in (1, 2, 3))
ok_off = all(bool(Gt[i, j].expr().simplify_full() == 0) for i in range(4) for j in range(4) if i != j)
print("[PASS] FLRW Einstein tensor: G_tt = 3(adot^2 + k)/a^2, G_ij = -(2 a addot + adot^2 + k) gamma_ij, off-diagonal zero (SageManifolds):", ok_tt and ok_ii and ok_off)
print("[PASS] scalar curvature R = -6 (a addot + adot^2 + k)/a^2 (negative for de Sitter in this signature):", bool((Rs.expr() + 6*(a*add + ad^2 + k)/a^2).simplify_full() == 0))
# perfect fluid at rest: T_tt = rho, T_ij = -p g_ij; Einstein equations G = 8 pi G T
GN = var("G_N"); rho = function("rho")(t); p = function("p")(t)
fr1 = 3*(ad^2 + k)/a^2 - 8*pi*GN*rho                       # tt equation
fr2 = -(2*a*add + ad^2 + k) - 8*pi*GN*p*a^2                # ij equation divided by gamma_ij
# second Friedmann equation: addot/a = -(4 pi G/3)(rho + 3p), using the first
acc = (fr2/a^2 + fr1/3)   # = -2 addot/a - 8 pi G (p + rho/3)
print("[PASS] the ij equation minus the tt equation gives addot/a = -(4 pi G/3)(rho + 3p):", bool((acc - (-2*add/a - 8*pi*GN*(p + rho/3))).simplify_full() == 0))
# continuity: d/dt (a^3 (fr1 terms)) combination; solve the first for rho and the second for p and check rho' + 3H(rho + p) = 0
rho_s = 3*(ad^2 + k)/(8*pi*GN*a^2); p_s = -(2*a*add + ad^2 + k)/(8*pi*GN*a^2)
print("[PASS] the two Friedmann equations imply the continuity equation rhodot + 3 (adot/a)(rho + p) = 0:", bool((diff(rho_s, t) + 3*ad/a*(rho_s + p_s)).simplify_full() == 0))
# flat solutions
tt, H, C = var("tt H C"); assume(tt > 0)
def check(aa, w):
    adv = diff(aa, tt); addv = diff(aa, tt, 2)
    rho_v = 3*adv^2/(8*pi*GN*aa^2); p_v = -(2*aa*addv + adv^2)/(8*pi*GN*aa^2)
    return bool((p_v - w*rho_v).simplify_full() == 0)
print("[PASS] for k = 0: a ~ t^(2/3) solves the equations with p = 0 (matter), a ~ t^(1/2) with p = rho/3 (radiation), a ~ e^(Ht) with p = -rho (cosmological constant):",
      check(C*tt^(2/3), 0) and check(C*tt^(1/2), 1/3) and check(C*exp(H*tt), -1))
# matter + Lambda: a = (Om/OL)^(1/3) sinh(3/2 sqrt(OL) H0 t)^(2/3) solves H^2 = H0^2 (Om a^-3 + OL)
Om, H0 = var("Om H0"); assume(Om > 0, Om < 1, H0 > 0, tt > 0); OL = 1 - Om
aL = (Om/OL)^(1/3)*sinh(3/2*sqrt(OL)*H0*tt)^(2/3)
print("[PASS] a = (Om/OL)^(1/3) sinh^(2/3)(3 sqrt(OL) H0 t/2) solves adot^2/a^2 = H0^2 (Om/a^3 + OL) with OL = 1 - Om:",
      bool((diff(aL, tt)^2/aL^2 - H0^2*(Om/aL^3 + OL)).simplify_full() == 0))
t0 = lambda om: float(2/(3*sqrt(1 - om))*arcsinh(sqrt((1 - om)/om)))
print("[PASS] the age of the universe is H0 t0 = 2/3 for matter alone and 0.964 for Om = 0.3, OL = 0.7:", abs(float(limit(2/(3*sqrt(1 - Om))*arcsinh(sqrt((1 - Om)/Om)), Om=1, dir="-")) - 2/3) < 1e-12 and abs(t0(0.3) - 0.964) < 5e-4, "(%.4f)" % t0(0.3))
# ---- static figure ----
taus = np.linspace(-1.2, 1.5, 600)
def lcdm(om, tau):
    T = t0(om) + tau
    return (om/(1 - om))**(1/3)*np.sinh(1.5*np.sqrt(1 - om)*T)**(2/3) if T > 0 else float("nan")
curves = [("radiation $a = (1 + 2\\tau)^{1/2}$", "darkorange", lambda q: (1 + 2*q)**0.5 if q > -0.5 else float("nan")),
          ("matter $a = (1 + 3\\tau/2)^{2/3}$", "royalblue", lambda q: (1 + 1.5*q)**(2/3) if q > -2/3 else float("nan")),
          ("$\\Lambda$: $a = e^{\\tau}$", "firebrick", lambda q: float(np.exp(q))),
          ("$\\Omega_m = 0.3$, $\\Omega_\\Lambda = 0.7$", "black", lambda q: lcdm(0.3, q))]
P = Graphics()
for lab, col, fn in curves:
    pts = [(float(q), float(fn(q))) for q in taus if not np.isnan(fn(q))]
    P += line(pts, color=col, thickness=2, legend_label=lab)
P += point((0, 1), size=40, color="black") + text("today", (0.05, 0.92), horizontal_alignment="left", fontsize=9)
P.axes_labels(["$\\tau$", "$a/a_0$"]); P.set_axes_range(-1.2, 1.5, 0, 3.5); P.set_legend_options(loc="upper left", font_size=8)
P.save(OUT, figsize=[7, 4.2])
print("wrote", os.path.relpath(OUT))
# ---- interactive: slider over Omega_Lambda (flat, Omega_m = 1 - Omega_Lambda) ----
from _interactive import plotly_html, _clean
import json
tau_i = [float(q) for q in np.linspace(-1.2, 2.0, 400)]
def series(ol):
    om = 1 - ol
    if ol == 0: return [(1 + 1.5*q)**(2/3) if q > -2/3 else None for q in tau_i]
    ys = []
    for q in tau_i:
        v = lcdm(om, q); ys.append(None if np.isnan(v) else float(v))
    return ys
ols = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
frames, steps = [], []
for ol in ols:
    frames.append({"name": "%.2f" % ol, "data": [{"x": tau_i, "y": series(ol)}], "traces": [0]})
    steps.append({"label": "%.2f" % ol, "method": "animate", "args": [["%.2f" % ol], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]})
traces = [{"x": tau_i, "y": series(0.7), "mode": "lines", "name": "a(t), flat, Omega_m = 1 - Omega_Lambda", "line": {"color": "black", "width": 3}},
          {"x": tau_i, "y": [(1 + 1.5*q)**(2/3) if q > -2/3 else None for q in tau_i], "mode": "lines", "name": "matter only", "line": {"color": "royalblue", "dash": "dash"}},
          {"x": tau_i, "y": [float(np.exp(q)) for q in tau_i], "mode": "lines", "name": "Lambda only", "line": {"color": "firebrick", "dash": "dash"}}]
layout = {"title": "Scale factor of a flat universe with matter and a cosmological constant (slider: Omega_Lambda)",
          "xaxis": {"title": "H0 (t - t0)", "range": [-1.2, 2.0]}, "yaxis": {"title": "a / a(t0)", "range": [0, 5]},
          "sliders": [{"active": 7, "currentvalue": {"prefix": "Omega_Lambda = "}, "steps": steps}], "legend": {"orientation": "h"}}
_path = plotly_html("flrw_scale_factor", traces, layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(_path, "w").write(_html)
print("wrote figures/interactive/flrw_scale_factor.html")
