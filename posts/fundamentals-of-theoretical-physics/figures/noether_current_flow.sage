# Figure: a conserved current. For the complex scalar wave packet phi = exp(-(x - vt)^2) exp(i(kx - wt)) in 1+1
# dimensions (signature (+, -)), the charge density j^0 (top, at three times) moves with the packet while its integral,
# the charge Q, stays constant; the current j^1 (bottom) is the flux that carries it, and d_t j^0 + d_x j^1 = 0 on the
# Klein-Gordon equation. Output: figures/noether_current_flow.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "noether_current_flow.svg")
x, t = var("x t", domain="real")
k, m = 2.0, 1.0; w = sqrt(k**2 + m**2); v = k / w
phi = exp(-(x - v * t)**2) * exp(I * (k * x - w * t)); phib = phi.conjugate()
j0 = (I * (phib * diff(phi, t) - phi * diff(phib, t))).simplify_full().real()      # charge density j^0 = j_t (eta^{tt} = 1)
j1 = (-I * (phib * diff(phi, x) - phi * diff(phib, x))).simplify_full().real()     # flux j^1 = -j_x: raising the index flips the sign (eta^{xx} = -1)
top = Graphics(); bottom = Graphics()
for i, tv in enumerate([0, 1.5, 3.0]):
    top += plot(j0.subs(t=tv), (x, -3, 7), color=hue(0.6 + 0.12 * i), thickness=1.5, legend_label=r"$t=%s$" % tv)
    bottom += plot(j1.subs(t=tv), (x, -3, 7), color=hue(0.6 + 0.12 * i), thickness=1.5)
top += text(r"charge density $j^0$ of a wave packet: it moves, its integral $Q$ does not change", (2, 5.2), fontsize=9)
bottom += text(r"current $j^1$: the flux carrying the charge", (2, 4.6), fontsize=9)
top.axes_labels([r"$x$", r"$j^0$"]); bottom.axes_labels([r"$x$", r"$j^1$"]); top.set_legend_options(loc="upper left", font_size=8)
g = graphics_array([[top], [bottom]]); fig = g.matplotlib(figsize=[8, 6]); fig.savefig(OUT, bbox_inches="tight")
Q = [numerical_integral(j0.subs(t=tv), -20, 25)[0] for tv in [0, 1.5, 3.0]]
print("[PASS] the charge Q = int j^0 dx is the same at the three times (to 1e-6):", max(Q) - min(Q) < 1e-6, [round(q, 6) for q in Q])
u, vv = function("u")(t, x), function("v")(t, x)
jg0 = 2 * (u * diff(vv, t) - vv * diff(u, t)); jg1 = 2 * (u * diff(vv, x) - vv * diff(u, x))
div = (diff(jg0, t) - diff(jg1, x)).simplify_full()               # d_mu j^mu = d_t j_t - d_x j_x with eta = diag(1, -1), from the lower-index components
box = lambda f: diff(f, t, 2) - diff(f, x, 2)                      # Box = d_t^2 - d_x^2
print("[PASS] the divergence of j = 2(u dv - v du) is 2(u Box v - v Box u), vanishing on the Klein-Gordon equation:", bool((div - 2 * (u * box(vv) - vv * box(u))).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
# interactive version: a time slider; the real part of phi (grey), the density j^0 (blue) and the current j^1 (red)
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
fj0 = fast_callable(j0, vars=[t, x], domain=RDF); fj1 = fast_callable(j1, vars=[t, x], domain=RDF)
fre = fast_callable(phi.real(), vars=[t, x], domain=RDF)
xs = [-3 + 10 * k / 300 for k in range(301)]; ts = [round(0.25 * k, 2) for k in range(17)]
_traces = []; steps = []
for i, tv in enumerate(ts):
    vis0 = i == 4
    _traces.append({"x": xs, "y": [fre(tv, a) for a in xs], "mode": "lines", "name": "Re phi", "line": {"color": "grey", "width": 1}, "visible": vis0})
    _traces.append({"x": xs, "y": [fj0(tv, a) for a in xs], "mode": "lines", "name": "j^0", "line": {"color": "royalblue", "width": 2.5}, "visible": vis0})
    _traces.append({"x": xs, "y": [fj1(tv, a) for a in xs], "mode": "lines", "name": "j^1", "line": {"color": "firebrick", "width": 2.5}, "xaxis": "x2", "yaxis": "y2", "visible": vis0})
for i, tv in enumerate(ts):
    vis = [False] * (3 * len(ts)); vis[3 * i] = vis[3 * i + 1] = vis[3 * i + 2] = True
    Qt = numerical_integral(j0.subs(t=tv), -20, 25)[0]
    steps.append({"method": "update", "label": "%.2f" % tv, "args": [{"visible": vis}, {"title": "t = %.2f:  Q = int j^0 dx = %.6f" % (tv, Qt)}]})
plotly_html("noether_current_flow", _traces, {"title": "t = 1.00:  Q = int j^0 dx = %.6f" % numerical_integral(j0.subs(t=1), -20, 25)[0],
            "xaxis": {"title": "", "domain": [0, 1], "anchor": "y", "range": [-3, 7]}, "yaxis": {"title": "Re phi, j^0", "domain": [0.55, 1], "range": [-1.2, 5.5]},
            "xaxis2": {"title": "x", "domain": [0, 1], "anchor": "y2", "range": [-3, 7]}, "yaxis2": {"title": "j^1", "domain": [0, 0.42], "range": [-1, 5]},
            "legend": {"orientation": "h"}, "sliders": [{"active": 4, "currentvalue": {"prefix": "t = "}, "pad": {"t": 40}, "steps": steps}]})
print("wrote figures/interactive/noether_current_flow.html")
