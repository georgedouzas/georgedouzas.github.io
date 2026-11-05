# Figure: a free Gaussian wave packet of initial width sigma and mean momentum p0 moves with velocity p0/m and spreads,
# its width growing as sigma(t) = sigma sqrt(1 + (hbar t/(2 m sigma^2))^2). Plot in units hbar = m = sigma = 1 with
# p0 = 2: the density |psi(x, t)|^2 and Re psi at t = 0, 1, 2, 4. The checks keep hbar, m, sigma, p0 symbolic.
# Output: figures/qm_wave_packet.svg (and an interactive time slider).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm_wave_packet.svg")
x, t, p0 = var("x t p0", domain="real")
hbar, m, s = var("hbar m s", domain="positive")
a = 1 + I * hbar * t / (2 * m * s ** 2)
psi = (2 * pi * s ** 2) ** (-1 / 4) * a ** (-1 / 2) * exp(-(x - p0 * t / m) ** 2 / (4 * s ** 2 * a) + I * p0 * x / hbar - I * p0 ** 2 * t / (2 * m * hbar))
res = (I * hbar * diff(psi, t) + hbar ** 2 / (2 * m) * diff(psi, x, 2)) / psi
print("[PASS] the spreading Gaussian packet solves the free Schroedinger equation i hbar psi_t = -(hbar^2/2m) psi_xx:", bool(res.simplify_full() == 0))
print("[PASS] at t = 0 it reduces to the Gaussian (2 pi sigma^2)^(-1/4) exp(-x^2/(4 sigma^2) + i p0 x/hbar):", bool((psi.subs(t=0) - (2 * pi * s ** 2) ** (-1 / 4) * exp(-x ** 2 / (4 * s ** 2) + I * p0 * x / hbar)).simplify_full() == 0))
st2 = s ** 2 * (1 + (hbar * t / (2 * m * s ** 2)) ** 2)
rho_formula = exp(-(x - p0 * t / m) ** 2 / (2 * st2)) / sqrt(2 * pi * st2)
ok = True
for vals in [{hbar: 1, m: 1, s: 1, p0: 2, t: 3 / 2, x: 1}, {hbar: 21 / 20, m: 7 / 10, s: 4 / 5, p0: 13 / 10, t: 2, x: 3}, {hbar: 1 / 3, m: 2, s: 1 / 2, p0: -1, t: 5, x: -2}]:
    v1 = abs(CC(psi.subs(vals))) ** 2; v2 = RR(rho_formula.subs(vals))
    ok = ok and abs(v1 - v2) < 1e-12
print("[PASS] |psi(x, t)|^2 is the Gaussian of mean p0 t/m and width sigma(t) = sigma sqrt(1 + (hbar t/(2 m sigma^2))^2) (three parameter sets):", ok)
num = {hbar: 1, m: 1, s: 1, p0: 2}
rt = rho_formula.subs(num)
ok2 = True
for tt in [0, 1, 3]:
    r = rt.subs(t=tt)
    nrm = numerical_integral(r, -40, 40)[0]; mx = numerical_integral(x * r, -40, 40)[0]; vx = numerical_integral((x - 2 * tt) ** 2 * r, -40, 40)[0]
    ok2 = ok2 and abs(nrm - 1) < 1e-9 and abs(mx - 2 * tt) < 1e-9 and abs(vx - (1 + tt ** 2 / 4)) < 1e-9
print("[PASS] numerically (hbar = m = sigma = 1, p0 = 2): norm 1, <x>(t) = 2t, (Delta x)^2 = 1 + t^2/4 at t = 0, 1, 3:", ok2)
# the figure
xx = var("xx")
pn = psi.subs(num)
cols = ["firebrick", "royalblue", "darkgreen", "darkorange"]
g1 = Graphics(); g2 = Graphics()
for tt, c in zip([0, 1, 2, 4], cols):
    f = fast_callable(pn.subs(t=tt), vars=[x], domain=CDF)
    g1 += plot(lambda u: abs(f(u)) ** 2, (-4, 14), color=c, thickness=1.8, plot_points=400, legend_label=r"$t = %d$" % tt)
    g2 += plot(lambda u: f(u).real(), (-4, 14), color=c, thickness=1.0, plot_points=800, legend_label=r"$t = %d$" % tt)
g1.axes_labels([r"$x$", r"$|\psi(x,t)|^2$"]); g2.axes_labels([r"$x$", r"$\mathrm{Re}\,\psi(x,t)$"])
g1.set_legend_options(font_size=8); g2.set_legend_options(font_size=8)
G = graphics_array([g1, g2])
G.matplotlib(figsize=[10, 3.8]).savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, cmath, math
xs = [-4 + 18 * k / 600.0 for k in range(601)]
def fr(tt):
    A = 1 + 0.5j * tt
    vals = [(2 * math.pi) ** (-0.25) * A ** (-0.5) * cmath.exp(-(u - 2 * tt) ** 2 / (4 * A) + 2j * u - 2j * tt) for u in xs]
    return [abs(complex(z)) ** 2 for z in vals], [complex(z).real for z in vals]
frames, steps = [], []
for k in range(0, 41):
    tt = k / 10.0
    d, r = fr(tt); nm = "%.1f" % tt
    frames.append({"name": nm, "data": [{"x": xs, "y": d}, {"x": xs, "y": r}], "traces": [0, 1]})
    steps.append({"label": nm, "method": "animate", "args": [[nm], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
d, r = fr(0)
traces = [{"x": xs, "y": d, "mode": "lines", "line": {"color": "royalblue", "width": 2.5}, "name": "|psi|^2"},
          {"x": xs, "y": r, "mode": "lines", "line": {"color": "firebrick", "width": 1}, "name": "Re psi"}]
layout = {"title": "Free Gaussian packet (hbar = m = sigma = 1, p0 = 2): slide t", "xaxis": {"title": "x", "range": [-4, 14]}, "yaxis": {"range": [-0.7, 0.7]},
          "sliders": [{"active": 0, "currentvalue": {"prefix": "t = "}, "steps": steps, "pad": {"t": 30}}], "legend": {"orientation": "h"}}
path = plotly_html("qm_wave_packet", traces, layout)
html = open(path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(path, "w").write(html)
print("wrote figures/interactive/qm_wave_packet.html")
