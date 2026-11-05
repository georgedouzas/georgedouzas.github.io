# Figure: a spinor under rotations. The spinor (1, 0) (spin up along z) is rotated about the x-axis by the angle
# theta with the SU(2) element exp(-i theta sigma_1 / 2); the plot shows the real and imaginary parts of its two
# components as functions of theta: after a rotation by 2 pi the spinor is multiplied by -1 and returns to itself
# only after 4 pi, while the expectation value <sigma_3> (the direction of the spin, grey) has period 2 pi.
# Output: figures/spinor_rotation.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spinor_rotation.svg")
th = var("th", domain="real")
s1 = matrix(SR, [[0, 1], [1, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
U = cos(th / 2) * identity_matrix(SR, 2) - I * sin(th / 2) * s1
psi = U * vector(SR, [1, 0])
p = plot(psi[0].real(), (th, 0, 4 * pi), color="royalblue", thickness=2, legend_label=r"$\mathrm{Re}\,\psi_1=\cos(\theta/2)$")
p += plot(psi[1].imag(), (th, 0, 4 * pi), color="firebrick", thickness=2, legend_label=r"$\mathrm{Im}\,\psi_2=-\sin(\theta/2)$")
spin_z = (psi.conjugate() * (s3 * psi)).simplify_full()
p += plot(spin_z, (th, 0, 4 * pi), color="grey", linestyle="dashed", legend_label=r"$\langle\sigma_3\rangle=\cos\theta$")
for xv, lab in [(2 * pi, r"$2\pi$: $\psi\to-\psi$"), (4 * pi, r"$4\pi$: $\psi\to\psi$")]:
    p += line([(xv, -1.1), (xv, 1.1)], color="black", linestyle="dotted") + text(lab, (xv, 1.25), fontsize=9)
p.set_legend_options(loc="lower left", font_size=8); p.axes_labels([r"$\theta$", ""])
p.save(OUT, figsize=[7, 3.6])
print("[PASS] the rotated spinor at 2 pi is -psi and at 4 pi is psi:", bool(psi.subs(th=2 * pi) == -vector(SR, [1, 0])) and bool(psi.subs(th=4 * pi) == vector(SR, [1, 0])))
print("[PASS] <sigma_3> = cos(theta), periodic with period 2 pi:", bool((spin_z - cos(th)).trig_reduce().simplify_full() == 0))
print("wrote", os.path.relpath(OUT))

# Interactive version: slide the rotation angle theta from 0 to 4 pi. Left: the spinor components Re psi_1 =
# cos(theta/2) and Im psi_2 = -sin(theta/2) with a marker at theta. Right: in the (y, z)-plane, the spin direction
# (<sigma_2>, <sigma_3>) = (-sin theta, cos theta) turns at the full angle, while the point (Im psi_2, Re psi_1) =
# (-sin(theta/2), cos(theta/2)) turns at half the angle: at theta = 2 pi the spin is back but the spinor is -psi.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
_ts = [4 * math.pi * i / 400.0 for i in range(401)]
_cs = [2 * math.pi * i / 200.0 for i in range(201)]
def _data(a):
    return [{"x": [a, a], "y": [math.cos(a / 2), -math.sin(a / 2)]},
            {"x": [0, -math.sin(a)], "y": [0, math.cos(a)]},
            {"x": [0, -math.sin(a / 2)], "y": [0, math.cos(a / 2)]}]
_frames, _steps = [], []
for i in range(0, 81):
    a = 4 * math.pi * i / 80.0
    lab = "%.2f pi" % (4.0 * i / 80.0)
    _frames.append({"name": lab, "data": _data(a), "traces": [2, 3, 4]})
    _steps.append({"label": lab, "method": "animate", "args": [[lab], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_d0 = _data(0.0)
_traces = [{"x": _ts, "y": [math.cos(u / 2) for u in _ts], "mode": "lines", "name": "Re psi_1 = cos(theta/2)", "line": {"color": "royalblue", "width": 2}},
           {"x": _ts, "y": [-math.sin(u / 2) for u in _ts], "mode": "lines", "name": "Im psi_2 = -sin(theta/2)", "line": {"color": "firebrick", "width": 2}},
           dict(_d0[0], mode="markers", name="current theta", marker={"size": 10, "color": "black"}),
           dict(_d0[1], mode="lines+markers", name="spin (<sigma_2>, <sigma_3>)", line={"color": "grey", "width": 3}, xaxis="x2", yaxis="y2"),
           dict(_d0[2], mode="lines+markers", name="spinor (Im psi_2, Re psi_1)", line={"color": "royalblue", "width": 3}, xaxis="x2", yaxis="y2"),
           {"x": [math.cos(c) for c in _cs], "y": [math.sin(c) for c in _cs], "mode": "lines", "showlegend": False, "line": {"color": "lightgrey"}, "xaxis": "x2", "yaxis": "y2"}]
_layout = {"title": "Rotating the spinor (1, 0) about x by theta: the spin returns at 2 pi, the spinor only at 4 pi",
           "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
           "xaxis": {"title": "theta", "range": [0, 4 * math.pi]}, "yaxis": {"range": [-1.2, 1.2]},
           "xaxis2": {"title": "y", "range": [-1.3, 1.3], "scaleanchor": "y2"}, "yaxis2": {"title": "z", "range": [-1.3, 1.3]},
           "sliders": [{"active": 0, "currentvalue": {"prefix": "theta = "}, "steps": _steps, "pad": {"t": 30}}], "legend": {"orientation": "h", "y": -0.25}}
_path = plotly_html("spinor_rotation", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
_sig2 = (psi.conjugate() * (matrix(SR, [[0, -I], [I, 0]]) * psi)).simplify_full()
print("[PASS] <sigma_2> = -sin(theta) for the rotated spinor (the spin turns in the (y, z)-plane):", bool((_sig2 + sin(th)).trig_reduce().simplify_full() == 0))
print("wrote figures/interactive/spinor_rotation.html")
