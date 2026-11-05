# Figure: a real scalar field in one space dimension. Left: the plane wave phi = cos(k x - omega t) at three times,
# moving to the right at the phase velocity omega/k. Right: the dispersion relation omega^2 = k^2 + m^2 of the
# Klein-Gordon equation for m = 0 (massless, straight line) and m = 1 (massive, hyperbola): the field's modes are
# relativistic particles of mass m. Output: figures/plane_wave.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_wave.svg")
x, t, k, w, m = var("x t k w m", domain="real")
kv, mv = 2.0, 1.0; wv = sqrt(kv**2 + mv**2)
phi = cos(kv * x - wv * t)
left = Graphics()
for i, tv in enumerate([0, 0.4, 0.8]):
    left += plot(phi.subs(t=tv), (x, -3, 3), color=hue(0.6 + 0.12 * i), thickness=1.5, legend_label=r"$t=%s$" % tv)
left += text(r"$\phi=\cos(kx-\omega t)$ moves right at $\omega/k$", (0, 1.35), fontsize=9)
left.axes_labels([r"$x$", r"$\phi$"]); left.set_legend_options(loc="lower right", font_size=8)
right = plot(abs(k), (k, -3, 3), color="grey", linestyle="dashed", legend_label=r"$m=0$: $\omega=|k|$")
right += plot(sqrt(k**2 + 1), (k, -3, 3), color="firebrick", thickness=2, legend_label=r"$m=1$: $\omega=\sqrt{k^2+m^2}$")
right += text(r"dispersion relation $\omega^2=k^2+m^2$", (0, 3.6), fontsize=9)
right.axes_labels([r"$k$", r"$\omega$"]); right.set_legend_options(loc="lower right", font_size=8)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9, 3.8]); fig.savefig(OUT, bbox_inches="tight")
phig = cos(k * x - w * t)
KG = (diff(phig, t, 2) - diff(phig, x, 2) + m**2 * phig).simplify_full()        # (Box + m^2) phi with Box = d_t^2 - d_x^2, signature (+, -)
print("[PASS] the Klein-Gordon operator (Box + m^2) on the plane wave gives (k^2 + m^2 - omega^2) cos(kx - omega t):", bool((KG - (k**2 + m**2 - w**2) * phig).simplify_full() == 0))
print("[PASS] it vanishes exactly on the dispersion relation:", bool(KG.subs(w=sqrt(k**2 + m**2)).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))

# Interactive version: slide the mass m. Left: the plane wave cos(kx - omega t) with k = 2 at t = 0 and t = 1 (the
# crest moves by the phase velocity omega/k). Right: the dispersion relation omega = sqrt(k^2 + m^2), the massless
# line, and the point k = 2 with its phase velocity omega/k and group velocity k/omega (product 1).
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
_xs = [-3 + 6 * i / 300.0 for i in range(301)]
_ks = [-3 + 6 * i / 200.0 for i in range(201)]
_k0 = 2.0
def _data(mm):
    om = math.sqrt(_k0**2 + mm**2)
    return [{"x": _xs, "y": [math.cos(_k0 * u) for u in _xs]},
            {"x": _xs, "y": [math.cos(_k0 * u - om * 1.0) for u in _xs]},
            {"x": _ks, "y": [math.sqrt(q**2 + mm**2) for q in _ks]},
            {"x": [_k0], "y": [om], "text": ["v_phase = %.3f, v_group = %.3f" % (om / _k0, _k0 / om)]}]
_frames, _steps = [], []
for i in range(0, 31):
    mm = i / 10.0
    _frames.append({"name": "%.1f" % mm, "data": _data(mm), "traces": [0, 1, 2, 4]})
    _steps.append({"label": "%.1f" % mm, "method": "animate", "args": [["%.1f" % mm], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_d0 = _data(1.0)
_traces = [dict(_d0[0], mode="lines", name="t = 0", line={"color": "royalblue", "width": 2}),
           dict(_d0[1], mode="lines", name="t = 1", line={"color": "firebrick", "width": 2}),
           dict(_d0[2], mode="lines", name="omega = sqrt(k^2 + m^2)", line={"color": "firebrick", "width": 2}, xaxis="x2", yaxis="y2"),
           {"x": _ks, "y": [abs(q) for q in _ks], "mode": "lines", "name": "m = 0: omega = |k|", "line": {"color": "grey", "dash": "dash"}, "xaxis": "x2", "yaxis": "y2"},
           dict(_d0[3], mode="markers+text", textposition="top left", name="k = 2", marker={"size": 10, "color": "black"}, xaxis="x2", yaxis="y2")]
_layout = {"title": "Klein-Gordon plane wave cos(kx - omega t), k = 2 (left); dispersion relation (right); slide m",
           "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
           "xaxis": {"title": "x", "range": [-3, 3]}, "yaxis": {"title": "phi", "range": [-1.2, 1.2]},
           "xaxis2": {"title": "k", "range": [-3, 3]}, "yaxis2": {"title": "omega", "range": [0, 4.5]},
           "sliders": [{"active": 10, "currentvalue": {"prefix": "m = "}, "steps": _steps, "pad": {"t": 30}}], "legend": {"orientation": "h", "y": -0.25}}
_path = plotly_html("plane_wave", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
print("[PASS] at every slider mass the phase and group velocities at k = 2 multiply to 1:", all(abs(math.sqrt(4 + (i / 10.0)**2) / 2 * 2 / math.sqrt(4 + (i / 10.0)**2) - 1) < 1e-12 for i in range(31)))
print("wrote figures/interactive/plane_wave.html")
