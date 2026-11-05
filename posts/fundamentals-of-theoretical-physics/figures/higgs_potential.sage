# Figure: the Mexican-hat potential V = (lambda/4)(|phi|^2 - mu^2)^2 of the abelian Higgs model over the complex
# phi-plane, with the circle of vacua |phi| = mu and the radial (Higgs) and angular (Goldstone) directions at a
# vacuum. Output: figures/higgs_potential.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "higgs_potential.png")
x, y, t = var("x y t"); lam, mu = 1, 1
V = lam/4 * (x^2 + y^2 - mu^2)^2
scene = plot3d(V, (x, -1.6, 1.6), (y, -1.6, 1.6), color="lightsteelblue", opacity=0.75, mesh=False)
scene += parametric_plot3d((mu*cos(t), mu*sin(t), 0), (t, 0, 2*pi), color="firebrick", thickness=4)
p0 = vector([mu, 0, 0])
scene += arrow3d(p0, p0 + vector([0.5, 0, 0]), color="darkgreen", width=2)      # radial: the Higgs direction
scene += arrow3d(p0, p0 + vector([0, 0.5, 0]), color="royalblue", width=2)      # tangential: the Goldstone direction
scene += point3d(tuple(p0), size=16, color="black")
scene.save(OUT, viewer="tachyon", figsize=[7, 6], zoom=1.2)
r = var("r"); Vr = lam/4*(r^2 - mu^2)^2
print("[PASS] V has its minimum value 0 on the circle r = mu:", bool(Vr.subs(r=mu) == 0) and bool(diff(Vr, r).subs(r=mu) == 0) and bool(diff(Vr, r, 2).subs(r=mu) > 0))
h = var("h"); print("[PASS] expansion V(mu + h) = lam mu^2 h^2 + lam mu h^3 + lam/4 h^4:", bool((Vr.subs(r=mu + h) - (lam*mu^2*h^2 + lam*mu*h^3 + lam/4*h^4)).expand() == 0))
print("wrote", os.path.relpath(OUT))

# Interactive version: the potential V = (lam/4)(|phi|^2 - s)^2 over the complex phi-plane as a 3D surface, with a
# slider for s = mu^2 from -1 to 1. For s <= 0 the minimum is the single point phi = 0 (symmetric phase); for s > 0
# the origin becomes a local maximum and the minima form the circle |phi| = sqrt(s) (red), the circle of vacua.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
_R = [1.6 * i / 30.0 for i in range(31)]; _TH = [2 * math.pi * j / 48.0 for j in range(49)]
def _surf(s):
    X = [[round(r * math.cos(t), 3) for r in _R] for t in _TH]; Y = [[round(r * math.sin(t), 3) for r in _R] for t in _TH]
    Z = [[round(0.25 * (r * r - s) ** 2, 4) for r in _R] for t in _TH]
    return X, Y, Z
def _circle(s):
    rv = math.sqrt(s) if s > 0 else 0.0
    return {"x": [rv * math.cos(t) for t in _TH], "y": [rv * math.sin(t) for t in _TH], "z": [0.0] * len(_TH)}
_svals = [round(-1 + 0.1 * k, 1) for k in range(21)]
_frames, _steps = [], []
for s in _svals:
    X, Y, Z = _surf(s)
    _frames.append({"name": "%.1f" % s, "data": [{"x": X, "y": Y, "z": Z}, _circle(s)], "traces": [0, 1]})
    _steps.append({"label": "%.1f" % s, "method": "animate", "args": [["%.1f" % s], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]})
X, Y, Z = _surf(1.0)
_traces = [{"type": "surface", "x": X, "y": Y, "z": Z, "colorscale": "Blues", "opacity": 0.85, "showscale": False, "name": "V"},
           dict(_circle(1.0), type="scatter3d", mode="lines", line={"color": "firebrick", "width": 8}, name="vacua |phi| = mu")]
_layout = {"title": "Mexican hat V = (1/4)(|phi|^2 - mu^2)^2: slide mu^2 through zero",
           "scene": {"xaxis": {"title": "Re phi"}, "yaxis": {"title": "Im phi"}, "zaxis": {"title": "V", "range": [0, 1.2]}, "aspectmode": "manual", "aspectratio": {"x": 1, "y": 1, "z": 0.7}},
           "sliders": [{"active": 20, "currentvalue": {"prefix": "mu^2 = "}, "steps": _steps, "pad": {"t": 20}}]}
_path = plotly_html("higgs_potential", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
def _argmin_r(s):
    return min(_R, key=lambda r: 0.25 * (r * r - s) ** 2)
print("[PASS] in every slider frame the grid minimum of V sits at |phi| = sqrt(max(mu^2, 0)) to grid accuracy:", all(abs(_argmin_r(s) - math.sqrt(max(s, 0))) <= 0.054 for s in _svals))
print("wrote figures/interactive/higgs_potential.html")
