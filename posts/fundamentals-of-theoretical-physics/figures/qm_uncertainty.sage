# Figure: position and momentum densities of normalised Gaussian wave packets of widths sigma = 1/2, 1, 2 (units
# hbar = 1): a narrow packet in x is wide in p, and Delta x Delta p = hbar/2 for every sigma. Checks: normalisation,
# expectation values and widths of the Gaussian with hbar explicit, its momentum wave function, the uncertainty
# products of the box eigenstates, the spin-1/2 Bloch vector and the Robertson relation for sigma_x, sigma_y, and
# the trace obstruction to [X, P] = i hbar in finite dimension. Output: figures/qm_uncertainty.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm_uncertainty.svg")
x, p, x0, p0 = var("x p x0 p0", domain="real")
hbar, s, L = var("hbar s L", domain="positive")
assume(hbar > 0, s > 0, L > 0)
psi = (2 * pi * s ** 2) ** (-1 / 4) * exp(-(x - x0) ** 2 / (4 * s ** 2) + I * p0 * x / hbar)
rho = (psi * psi.conjugate()).simplify_full()
ex = lambda f: integrate(f, x, -oo, oo).simplify_full()
norm = ex(rho)
xm = ex(x * rho)
x2 = ex(x ** 2 * rho)
dpsi = diff(psi, x); d2psi = diff(psi, x, 2)
pm = ex((psi.conjugate() * (-I * hbar) * dpsi).simplify_full())
p2 = ex((psi.conjugate() * (-hbar ** 2) * d2psi).simplify_full())
dx2 = (x2 - xm ** 2).simplify_full(); dp2 = (p2 - pm ** 2).simplify_full()
print("[PASS] the Gaussian packet is normalised with <x> = x0, <p> = p0:", bool(norm == 1) and bool(xm == x0) and bool(pm == p0))
print("[PASS] the Gaussian packet has (Delta x)^2 = sigma^2, (Delta p)^2 = hbar^2/(4 sigma^2), Delta x Delta p = hbar/2:", bool(dx2 == s ** 2) and bool(dp2 == hbar ** 2 / (4 * s ** 2)) and bool(sqrt(dx2 * dp2).simplify_full() == hbar / 2))
psi00 = psi.subs(x0=0, p0=0)
phi = (integrate(exp(-I * p * x / hbar) * psi00, x, -oo, oo) / sqrt(2 * pi * hbar)).simplify_full()
phi_expected = (2 * s ** 2 / (pi * hbar ** 2)) ** (1 / 4) * exp(-s ** 2 * p ** 2 / hbar ** 2)
print("[PASS] the momentum wave function of the centred Gaussian is (2 sigma^2/(pi hbar^2))^(1/4) exp(-sigma^2 p^2/hbar^2), normalised:", bool((phi / phi_expected).canonicalize_radical() == 1) and bool(integrate(phi_expected ** 2, p, -oo, oo).simplify_full() == 1))
n = var("n"); assume(n, "integer"); assume(n > 0)
ok = True
for nn in range(1, 6):
    ph = sqrt(2 / L) * sin(nn * pi * x / L)
    a = lambda f: integrate(f, x, 0, L).simplify_full()
    mx = a(x * ph ** 2); vx = (a(x ** 2 * ph ** 2) - mx ** 2).simplify_full()
    vp = a(ph * (-hbar ** 2) * diff(ph, x, 2)).simplify_full()
    prod2 = (vx * vp).simplify_full()
    ok = ok and bool(mx == L / 2) and bool(prod2 == hbar ** 2 / 4 * (nn ** 2 * pi ** 2 / 3 - 2))
print("[PASS] the box eigenstates n = 1..5 have <x> = L/2, <p> = 0 and (Delta x Delta p)^2 = (hbar^2/4)(n^2 pi^2/3 - 2):", ok)
print("[PASS] the box ground state has Delta x Delta p = 0.5679 hbar > hbar/2:", bool(abs(sqrt(pi ** 2 / 3 - 2).n() / 2 - 0.5679) < 1e-4))
th, ph_ = var("th ph_", domain="real")
v = vector(SR, [cos(th / 2), exp(I * ph_) * sin(th / 2)])
sx = matrix(SR, [[0, 1], [1, 0]]); sy = matrix(SR, [[0, -I], [I, 0]]); sz = matrix(SR, [[1, 0], [0, -1]])
E = lambda A: (v.conjugate() * A * v).simplify_full()
bloch = [E(S).rectform().trig_reduce().simplify_full() for S in (sx, sy, sz)]
print("[PASS] the spin-1/2 state (cos(theta/2), e^(i phi) sin(theta/2)) has Bloch vector <sigma> = (sin th cos ph, sin th sin ph, cos th):", all(bool((b - c).trig_reduce().simplify_full() == 0) for b, c in zip(bloch, [sin(th) * cos(ph_), sin(th) * sin(ph_), cos(th)])))
nb = [sin(th) * cos(ph_), sin(th) * sin(ph_), cos(th)]
lhs = (1 - nb[0] ** 2) * (1 - nb[1] ** 2) - nb[2] ** 2
print("[PASS] Robertson for sigma_x, sigma_y: (Delta sx)^2 (Delta sy)^2 - <sz>^2 = sin^4 th cos^2 ph sin^2 ph >= 0:", bool((lhs - sin(th) ** 4 * cos(ph_) ** 2 * sin(ph_) ** 2).trig_reduce().simplify_full() == 0))
X = random_matrix(QQ, 4, 4); P = random_matrix(QQ, 4, 4)
print("[PASS] tr[X, P] = 0 for matrices, so [X, P] = i hbar 1 (trace 4 i hbar) has no finite-dimensional solution:", (X * P - P * X).trace() == 0)
# the figure
xx = var("xx")
cols = ["firebrick", "royalblue", "darkgreen"]
g1 = Graphics(); g2 = Graphics()
for sg, c in zip([1 / 2, 1, 2], cols):
    g1 += plot(exp(-xx ** 2 / (2 * sg ** 2)) / sqrt(2 * pi * sg ** 2), (xx, -5, 5), color=c, thickness=1.8, legend_label=r"$\sigma = %s$" % latex(sg))
    g2 += plot(sqrt(2 * sg ** 2 / pi) * exp(-2 * sg ** 2 * xx ** 2), (xx, -3, 3), color=c, thickness=1.8, legend_label=r"$\Delta p = %s$" % latex(1 / (2 * sg)))
g1.axes_labels([r"$x$", r"$|\psi(x)|^2$"]); g2.axes_labels([r"$p$", r"$|\tilde{\psi}(p)|^2$"])
G = graphics_array([g1, g2])
G.matplotlib(figsize=[10, 3.8]).savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
xs = [-5 + 10 * k / 300.0 for k in range(301)]; ps = [-3 + 6 * k / 300.0 for k in range(301)]
def dens(sg):
    return ([math.exp(-u ** 2 / (2 * sg ** 2)) / math.sqrt(2 * math.pi * sg ** 2) for u in xs],
            [math.sqrt(2 * sg ** 2 / math.pi) * math.exp(-2 * sg ** 2 * u ** 2) for u in ps])
frames, steps = [], []
for k in range(0, 31):
    sg = 0.3 + 0.1 * k
    d1, d2 = dens(sg)
    nm = "%.1f" % sg
    frames.append({"name": nm, "data": [{"x": xs, "y": d1}, {"x": ps, "y": d2}], "traces": [0, 1]})
    steps.append({"label": nm, "method": "animate", "args": [[nm], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
d1, d2 = dens(1.0)
traces = [{"x": xs, "y": d1, "mode": "lines", "line": {"color": "royalblue", "width": 2}, "name": "|psi(x)|^2"},
          {"x": ps, "y": d2, "mode": "lines", "line": {"color": "firebrick", "width": 2}, "name": "|psi~(p)|^2", "xaxis": "x2", "yaxis": "y2"}]
layout = {"title": "Gaussian packet (hbar = 1): slide sigma; Delta x Delta p = 1/2 throughout", "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
          "xaxis": {"title": "x", "range": [-5, 5]}, "yaxis": {"range": [0, 1.4]}, "xaxis2": {"title": "p", "range": [-3, 3]}, "yaxis2": {"range": [0, 2.6]},
          "sliders": [{"active": 7, "currentvalue": {"prefix": "sigma = "}, "steps": steps, "pad": {"t": 30}}], "legend": {"orientation": "h"}}
path = plotly_html("qm_uncertainty", traces, layout)
html = open(path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(path, "w").write(html)
print("wrote figures/interactive/qm_uncertainty.html")
