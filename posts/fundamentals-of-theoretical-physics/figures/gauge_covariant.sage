# Figure: why a local phase rotation needs a gauge field. Top: a complex field phi = e^{i theta(x)} on a line, drawn
# as a phase arrow at each point, with a position-dependent rotation alpha(x) applied (right); the ordinary
# derivative compares arrows in different frames. Bottom: the gauge potential A(x) = -alpha'(x) that compensates
# the rotation, so that D phi = (d + i A) phi transforms like phi. Output: figures/gauge_covariant.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauge_covariant.svg")
x = var("x", domain="real")
theta = 0.4 * x; alpha = 1.2 * sin(x)
top = Graphics()
for k, xv in enumerate(srange(0, 6.3, 0.5)):
    th = float(theta.subs(x=xv)); al = float(alpha.subs(x=xv))
    top += arrow((xv, 0), (xv + 0.45 * cos(th), 0.45 * sin(th)), color="royalblue", width=1)
    top += arrow((xv, -1.6), (xv + 0.45 * cos(th + al), -1.6 + 0.45 * sin(th + al)), color="firebrick", width=1)
top += text(r"$\phi(x)=e^{i\theta(x)}$: phase arrows", (3.1, 0.75), fontsize=9, color="royalblue")
top += text(r"$e^{i\alpha(x)}\phi(x)$: rotated by a different angle at each point", (3.1, -0.85), fontsize=9, color="firebrick")
top.axes(False); top.set_aspect_ratio(1)
bottom = plot(alpha, (x, 0, 6.3), color="firebrick", thickness=1.5, legend_label=r"$\alpha(x)$")
bottom += plot(-diff(alpha, x), (x, 0, 6.3), color="darkgreen", thickness=1.5, legend_label=r"$\delta A=-\alpha'(x)$")
bottom += text(r"the shift of the gauge field that undoes the extra term $i\alpha'\phi$ in $\partial(e^{i\alpha}\phi)$", (3.15, 1.9), fontsize=9)
bottom.set_legend_options(loc="lower left", font_size=8); bottom.axes_labels([r"$x$", ""])
g = graphics_array([[top], [bottom]]); fig = g.matplotlib(figsize=[8, 5.5]); fig.savefig(OUT, bbox_inches="tight")
A = function("A")(x); phi = exp(I * theta); al = function("al")(x)
D = lambda f, a: diff(f, x) + I * a * f
lhs = D(exp(I * al) * phi, A - diff(al, x)); rhs = exp(I * al) * D(phi, A)
print("[PASS] D phi = (d + i A) phi transforms covariantly, D'(e^{i alpha} phi) = e^{i alpha} D phi, when A -> A - alpha':", bool((lhs - rhs).simplify_full() == 0))
print("[PASS] the ordinary derivative does not: d(e^{i alpha} phi) - e^{i alpha} d phi = i alpha' e^{i alpha} phi:", bool((diff(exp(I * al) * phi, x) - exp(I * al) * diff(phi, x) - I * diff(al, x) * exp(I * al) * phi).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))

# Interactive version: the gauge-transformed field configuration. phi(x) = (1 + 0.3 cos x) e^{0.4 i x} with the gauge
# field A(x) = 0.3 cos x is transformed by alpha(x) = a sin x (slider a). Top: the phase arrows of e^{i alpha} phi.
# Bottom: |d phi'| changes with a, while |D' phi'| = |(d + i A') phi'| with A' = A - alpha' stays the same curve.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math, cmath
_xs = [6.3 * i / 300.0 for i in range(301)]; _xa = [0.5 * k for k in range(13)]
_phi = lambda u: (1 + 0.3 * math.cos(u)) * cmath.exp(0.4j * u)
_dphi = lambda u: (-0.3 * math.sin(u) + 0.4j * (1 + 0.3 * math.cos(u))) * cmath.exp(0.4j * u)
_A = lambda u: 0.3 * math.cos(u)
def _data(a):
    al = lambda u: a * math.sin(u); dal = lambda u: a * math.cos(u)
    ph2 = lambda u: cmath.exp(1j * al(u)) * _phi(u)
    dph2 = lambda u: cmath.exp(1j * al(u)) * (_dphi(u) + 1j * dal(u) * _phi(u))
    A2 = lambda u: _A(u) - dal(u)
    ax, ay = [], []
    for u in _xa:
        z = 0.4 * ph2(u); ax += [u, u + z.real, None]; ay += [0, z.imag, None]
    return [{"x": ax, "y": ay},
            {"x": _xs, "y": [abs(dph2(u)) for u in _xs]},
            {"x": _xs, "y": [abs(dph2(u) + 1j * A2(u) * ph2(u)) for u in _xs]},
            {"x": _xs, "y": [A2(u) for u in _xs]}]
_avals = [round(0.1 * k, 1) for k in range(16)]
_frames, _steps = [], []
for a in _avals:
    _frames.append({"name": "%.1f" % a, "data": _data(a), "traces": [0, 1, 2, 3]})
    _steps.append({"label": "%.1f" % a, "method": "animate", "args": [["%.1f" % a], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_d0 = _data(0.0)
_traces = [dict(_d0[0], mode="lines", name="phase arrows of e^{i alpha} phi", line={"color": "firebrick", "width": 2}),
           dict(_d0[1], mode="lines", name="|d phi'| (changes)", line={"color": "darkorange", "width": 2}, xaxis="x2", yaxis="y2"),
           dict(_d0[2], mode="lines", name="|D' phi'| (gauge invariant)", line={"color": "royalblue", "width": 3}, xaxis="x2", yaxis="y2"),
           dict(_d0[3], mode="lines", name="A' = A - alpha'", line={"color": "darkgreen", "width": 2, "dash": "dot"}, xaxis="x2", yaxis="y2")]
_layout = {"title": "A local phase rotation alpha(x) = a sin x and the compensating gauge field; slide a",
           "grid": {"rows": 2, "columns": 1, "pattern": "independent"},
           "xaxis": {"range": [-0.3, 6.6]}, "yaxis": {"range": [-0.6, 0.6], "scaleanchor": "x"},
           "xaxis2": {"title": "x", "range": [0, 6.3]}, "yaxis2": {"range": [-1.8, 2.2]},
           "sliders": [{"active": 0, "currentvalue": {"prefix": "a = "}, "steps": _steps, "pad": {"t": 40}}], "legend": {"orientation": "h", "y": -0.2}}
_path = plotly_html("gauge_covariant", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
_ref = _data(0.0)[2]["y"]
print("[PASS] in every slider frame |D' phi'| equals |D phi| pointwise (covariance), while |d phi'| changes:", all(max(abs(p - q) for p, q in zip(f["data"][2]["y"], _ref)) < 1e-12 for f in _frames) and max(abs(p - q) for p, q in zip(_frames[-1]["data"][1]["y"], _frames[0]["data"][1]["y"])) > 0.5)
print("wrote figures/interactive/gauge_covariant.html")
