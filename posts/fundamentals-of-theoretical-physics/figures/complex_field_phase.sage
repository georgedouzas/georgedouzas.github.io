# Figure: a complex scalar field and its phase rotation. Left: the value phi = u + i v at a point is a vector in the
# complex plane; the global U(1) symmetry phi -> e^{i alpha} phi rotates it (arrows), preserving |phi|. Right: the
# conserved U(1) current j^mu = i (phi-bar d phi - phi d phi-bar) for a plane wave phi = e^{i(kx - omega t)} is the
# constant vector (2 omega, 2 k): charge density and charge flux, proportional to the wave's momentum.
# Output: figures/complex_field_phase.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "complex_field_phase.svg")
left = circle((0, 0), 1.3, color="lightgrey")
for i, a in enumerate([0, 0.5, 1.0, 1.5]):
    z = 1.3 * exp(I * (0.3 + a))
    left += arrow((0, 0), (z.real(), z.imag()), color=hue(0.55 + 0.1 * i), width=1.5)
    left += text(r"$e^{i\alpha}\phi$, $\alpha=%s$" % a, (1.55 * z.real(), 1.55 * z.imag()), fontsize=8)
left += text(r"$\phi=u+iv$: the phase rotation preserves $|\phi|$", (0, 2.3), fontsize=9)
left.axes_labels([r"$u$", r"$v$"]); left.set_aspect_ratio(1)
x, t, k, w = var("x t k w", domain="real")
phi = exp(I * (k * x - w * t)); phib = phi.conjugate()
j0 = (I * (phib * diff(phi, t) - phi * diff(phib, t))).simplify_full()
j1 = (I * (phib * diff(phi, x) - phi * diff(phib, x))).simplify_full()
right = Graphics()
for xv in [-2, -1, 0, 1, 2]:
    for tv in [0, 1, 2]:
        right += arrow((xv, tv), (xv + 0.25 * 1.5, tv + 0.25 * 2.0), color="firebrick", width=1)
right += text(r"$j^\mu\propto(\omega, k)$ for $\phi=e^{i(kx-\omega t)}$: a uniform charge flow along the wave", (0, 3.2), fontsize=9)
right.axes_labels([r"$x$", r"$t$"]); right.set_aspect_ratio(1)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9, 4]); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] the lower-index current j_mu = i(phib d_mu phi - phi d_mu phib) of the plane wave is (j_t, j_x) = (2 omega, -2 k), proportional to (omega, -k):", bool((j0 - 2 * w).simplify_full() == 0) and bool((j1 + 2 * k).simplify_full() == 0))
u, v, al = var("u v al", domain="real")
rot = exp(I * al) * (u + I * v)
print("[PASS] the phase rotation preserves |phi|^2:", bool((rot * rot.conjugate() - (u**2 + v**2)).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))

# Interactive version: slide the angle alpha of the global phase rotation. Left: phi = 1.3 e^{0.3 i} and its rotation
# e^{i alpha} phi on the circle |phi| = 1.3. Right: the real and imaginary parts of the field e^{i alpha} e^{i k x}
# (k = 2) along x, which shift along the axis while |phi| stays constant.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
_xs = [-math.pi + 2 * math.pi * i / 300.0 for i in range(301)]
_cs = [2 * math.pi * i / 200.0 for i in range(201)]
def _data(a):
    z = 1.3 * complex(math.cos(0.3 + a), math.sin(0.3 + a))
    return [{"x": [0, z.real], "y": [0, z.imag]},
            {"x": _xs, "y": [math.cos(2 * u + a) for u in _xs]},
            {"x": _xs, "y": [math.sin(2 * u + a) for u in _xs]}]
_frames, _steps = [], []
for i in range(0, 64):
    a = i / 10.0
    _frames.append({"name": "%.1f" % a, "data": _data(a), "traces": [1, 2, 3]})
    _steps.append({"label": "%.1f" % a, "method": "animate", "args": [["%.1f" % a], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_d0 = _data(0.0)
_traces = [{"x": [1.3 * math.cos(c) for c in _cs], "y": [1.3 * math.sin(c) for c in _cs], "mode": "lines", "name": "|phi| = 1.3", "line": {"color": "lightgrey"}},
           dict(_d0[0], mode="lines+markers", name="e^{i alpha} phi", line={"color": "royalblue", "width": 3}),
           dict(_d0[1], mode="lines", name="Re e^{i alpha} e^{2ix}", line={"color": "royalblue", "width": 2}, xaxis="x2", yaxis="y2"),
           dict(_d0[2], mode="lines", name="Im e^{i alpha} e^{2ix}", line={"color": "firebrick", "width": 2}, xaxis="x2", yaxis="y2"),
           {"x": [0, 1.3 * math.cos(0.3)], "y": [0, 1.3 * math.sin(0.3)], "mode": "lines", "name": "phi (alpha = 0)", "line": {"color": "grey", "dash": "dot"}}]
_layout = {"title": "Global U(1): phi -> e^{i alpha} phi at a point (left) and along a field (right); slide alpha",
           "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
           "xaxis": {"title": "u = Re phi", "range": [-1.6, 1.6], "scaleanchor": "y"}, "yaxis": {"title": "v = Im phi", "range": [-1.6, 1.6]},
           "xaxis2": {"title": "x", "range": [-math.pi, math.pi]}, "yaxis2": {"range": [-1.2, 1.2]},
           "sliders": [{"active": 0, "currentvalue": {"prefix": "alpha = "}, "steps": _steps, "pad": {"t": 30}}], "legend": {"orientation": "h", "y": -0.25}}
_path = plotly_html("complex_field_phase", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
print("[PASS] every slider frame keeps |e^{i alpha} phi| = 1.3:", all(abs(math.hypot(f["data"][0]["x"][1], f["data"][0]["y"][1]) - 1.3) < 1e-12 for f in _frames))
print("wrote figures/interactive/complex_field_phase.html")
