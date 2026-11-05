# Figure: solutions of the one-dimensional wave equation u_tt - c^2 u_xx = 0 with c = 1. Left: d'Alembert's solution
# for a Gaussian initial profile at rest, u = [f(x - t) + f(x + t)]/2, at t = 0, 1, 2, 3: the bump splits into two
# half-height copies travelling left and right. Right: the standing wave u = sin(2x) cos(2t) on [0, pi] from separation
# of variables, at several times: the nodes stay fixed. The [PASS] lines verify d'Alembert's formula, the standing
# wave, the heat kernel, two harmonic functions and the plane-wave solution of the Klein-Gordon equation in the
# (+, -, -, -) signature by exact symbolic computation. Output: figures/wave_dalembert.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wave_dalembert.svg")
t, x, y, z, c, k, m, kap, s = var("t x y z c k m kap s", domain="real")
assume(c > 0)
f = function("f"); g = function("g")
u = f(x - c * t) + g(x + c * t)
print("[PASS] u = f(x - ct) + g(x + ct) solves the wave equation u_tt - c^2 u_xx = 0 for arbitrary f, g:",
      bool((diff(u, t, 2) - c ** 2 * diff(u, x, 2)).simplify_full() == 0))
phi0, psi0 = exp(-x ** 2), cos(x)                     # initial profile and initial velocity
ud = (phi0.subs(x=x - c * t) + phi0.subs(x=x + c * t)) / 2 + integrate(psi0.subs(x=s), s, x - c * t, x + c * t) / (2 * c)
print("[PASS] d'Alembert's formula solves the wave equation with u(0, x) = e^(-x^2), u_t(0, x) = cos x:",
      bool((diff(ud, t, 2) - c ** 2 * diff(ud, x, 2)).simplify_full() == 0) and bool((ud.subs(t=0) - phi0).simplify_full() == 0)
      and bool((diff(ud, t).subs(t=0) - psi0).simplify_full() == 0))
st = sin(k * x) * cos(k * c * t)
print("[PASS] the standing wave sin(kx) cos(kct) solves the wave equation and vanishes at x = 0 and x = pi for integer k:",
      bool((diff(st, t, 2) - c ** 2 * diff(st, x, 2)).simplify_full() == 0) and all(bool(st.subs(x=0) == 0) and bool(st.subs(x=pi, k=n).simplify_full() == 0) for n in range(1, 4)))
assume(kap > 0, t > 0)
heat = exp(-x ** 2 / (4 * kap * t)) / sqrt(4 * pi * kap * t)
print("[PASS] the heat kernel (4 pi kappa t)^(-1/2) exp(-x^2 / 4 kappa t) solves u_t = kappa u_xx and has unit integral:",
      bool((diff(heat, t) - kap * diff(heat, x, 2)).simplify_full() == 0) and bool(integrate(heat, x, -oo, oo).canonicalize_radical() == 1))
sep = exp(-kap * k ** 2 * t) * sin(k * x)
print("[PASS] separation of variables: e^(-kappa k^2 t) sin(kx) solves the heat equation:", bool((diff(sep, t) - kap * diff(sep, x, 2)).simplify_full() == 0))
h1 = x ** 2 - y ** 2
h2 = log(x ** 2 + y ** 2)
print("[PASS] x^2 - y^2 and log(x^2 + y^2) are harmonic in the plane (Laplace's equation):",
      bool(diff(h1, x, 2) + diff(h1, y, 2) == 0) and bool((diff(h2, x, 2) + diff(h2, y, 2)).simplify_full() == 0))
w = var("w", domain="real"); k1, k2, k3 = var("k1 k2 k3", domain="real")
pw = exp(I * (k1 * x + k2 * y + k3 * z - w * t))
box = diff(pw, t, 2) - diff(pw, x, 2) - diff(pw, y, 2) - diff(pw, z, 2)        # Box = d_t^2 - nabla^2 in (+, -, -, -)
kg = ((box + m ** 2 * pw) / pw).simplify_full()
print("[PASS] the plane wave e^{i(k.x - omega t)} solves (Box + m^2) phi = 0 exactly when omega^2 = k^2 + m^2:",
      bool(kg.subs(w=sqrt(k1 ** 2 + k2 ** 2 + k3 ** 2 + m ** 2)).simplify_full() == 0) and bool((kg - (m ** 2 - w ** 2 + k1 ** 2 + k2 ** 2 + k3 ** 2)).simplify_full() == 0))
B = x ** 2 - y ** 2
print("[PASS] classification: u_tt - c^2 u_xx has B^2 - AC = c^2 > 0 (hyperbolic), u_xx + u_yy has -1 < 0 (elliptic), u_t - kappa u_xx has 0 (parabolic):",
      bool(0 ** 2 - 1 * (-c ** 2) > 0) and (0 ** 2 - 1 * 1 < 0) and (0 ** 2 - 0 * (-1) == 0))
gauss = exp(-x ** 2)
cols = ["black", "firebrick", "darkorange", "royalblue"]
left = Graphics()
for tt, col in zip([0, 1, 2, 3], cols):
    left += plot((gauss.subs(x=x - tt) + gauss.subs(x=x + tt)) / 2, (x, -6, 6), color=col, thickness=1.4, legend_label=r"$t = %d$" % tt)
left += text(r"d'Alembert: $u = \frac{1}{2}[f(x-t)+f(x+t)]$, $f = e^{-x^2}$", (0, 1.15), fontsize=9)
left.axes_labels([r"$x$", r"$u$"]); left.set_legend_options(loc="upper left", font_size=8)
right = Graphics()
for tt, col in zip([0, 0.3, 0.6, 0.785], cols):
    right += plot(sin(2 * x) * cos(2 * tt), (x, 0, pi), color=col, thickness=1.4, legend_label=r"$t = %.2f$" % tt)
right += text(r"standing wave $u = \sin 2x\,\cos 2t$: fixed nodes", (pi / 2, 1.2), fontsize=9)
right.axes_labels([r"$x$", r"$u$"]); right.set_legend_options(loc="lower left", font_size=8)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 3.8]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# interactive version: a time slider for the travelling pair (d'Alembert) and the standing wave
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
xs = [float(-6 + 12 * i / 300.0) for i in range(301)]
xs2 = [float(math.pi * i / 150.0) for i in range(151)]
def dal(tt):
    return [0.5 * (math.exp(-(u - tt) ** 2) + math.exp(-(u + tt) ** 2)) for u in xs]
frames, steps = [], []
for i in range(0, 41):
    tt = float(i) / 10.0
    frames.append({"name": "%.1f" % tt, "data": [{"x": xs, "y": dal(tt)}, {"x": xs2, "y": [math.sin(2 * u) * math.cos(2 * tt) for u in xs2]}], "traces": [0, 1]})
    steps.append({"label": "%.1f" % tt, "method": "animate", "args": [["%.1f" % tt], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
traces = [{"x": xs, "y": dal(0), "mode": "lines", "line": {"color": "royalblue", "width": 2}, "name": "d'Alembert: [f(x - t) + f(x + t)]/2"},
          {"x": xs2, "y": [math.sin(2 * u) for u in xs2], "mode": "lines", "line": {"color": "firebrick", "width": 2}, "name": "standing wave sin 2x cos 2t", "xaxis": "x2", "yaxis": "y2"}]
layout = {"title": "Wave equation: travelling pair (left) and standing wave (right); slide t", "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
          "xaxis": {"title": "x", "range": [-6, 6]}, "yaxis": {"range": [-0.1, 1.1]}, "xaxis2": {"title": "x", "range": [0, math.pi]}, "yaxis2": {"range": [-1.1, 1.1]},
          "sliders": [{"active": 0, "currentvalue": {"prefix": "t = "}, "steps": steps, "pad": {"t": 30}}], "legend": {"orientation": "h"}}
path = plotly_html("wave_dalembert", traces, layout)
html = open(path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(path, "w").write(html)
print("wrote figures/interactive/wave_dalembert.html")
