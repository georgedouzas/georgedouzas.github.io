# Figure: the spectrum of the abelian Higgs model before and after symmetry breaking. Left: with mu^2 < 0 in
# V = (lambda/4)(|phi|^2 - mu^2)^2 read as -m^2|phi|^2 + ..., the two real scalars have mass m and the photon is
# massless with two polarisations. Right: after breaking, one Higgs boson of mass sqrt(lambda) mu and one massive
# vector of mass sqrt(2) e mu with three polarisations; the Goldstone boson has been eaten. Degrees of freedom:
# 2 + 2 = 1 + 3. Output: figures/higgs_spectrum.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "higgs_spectrum.svg")
lam, mu, e = 1.0, 1.0, 0.8
m_h = sqrt(lam) * mu; m_A = sqrt(2) * e * mu
def bars(items, title):
    p = Graphics()
    for i, (name, mass, dof, col) in enumerate(items):
        p += polygon([(i - 0.3, 0), (i + 0.3, 0), (i + 0.3, mass), (i - 0.3, mass)], color=col, alpha=0.7)
        p += text("%s\n%d d.o.f." % (name, dof), (i, -0.25), fontsize=9)
        p += text("$m=%.2f$" % float(mass) if mass > 0 else "massless", (i, mass + 0.1), fontsize=8)
    p += text(title, (1, 1.75), fontsize=10)
    p.axes(False); p.xmin(-0.6); p.xmax(2.6); p.ymin(-0.5); p.ymax(1.9)   # identical frames so the panels align
    return p
left = bars([("scalar $u$", 0, 1, "royalblue"), ("scalar $v$", 0, 1, "royalblue"), ("photon", 0, 2, "firebrick")], r"symmetric phase: $2+2$ degrees of freedom")
right = bars([("Higgs $\\sigma$", m_h, 1, "royalblue"), ("massive vector", m_A, 3, "firebrick")], r"broken phase: $1+3$ degrees of freedom")
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9, 3.6], axes=False); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] the degrees of freedom are conserved, 1 + 1 + 2 = 1 + 3:", 1 + 1 + 2 == 1 + 3)
print("[PASS] m_h^2 = lambda mu^2 and m_A^2 = 2 e^2 mu^2 for the chosen parameters:", bool(m_h**2 == lam * mu**2) and bool(m_A**2 == 2 * e**2 * mu**2))
print("wrote", os.path.relpath(OUT))

# Interactive version: the masses of the abelian Higgs model as functions of the couplings, in units of mu. The Higgs
# mass m_sigma = sqrt(lam) mu is drawn against lam; the vector mass m_A = sqrt(2) e mu is a horizontal line set by the
# slider for e. The two are equal at lam = 2 e^2 (marker): below, the Higgs is the lighter particle, above the vector.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
_ls = [3.0 * i / 300.0 for i in range(301)]
_evals = [round(0.1 * k, 1) for k in range(1, 13)]
def _data(ev):
    mA = math.sqrt(2) * ev; lc = 2 * ev * ev
    return [{"x": [0, 3], "y": [mA, mA]}, {"x": [lc] if lc <= 3 else [], "y": [mA] if lc <= 3 else []}]
_frames, _steps = [], []
for ev in _evals:
    _frames.append({"name": "%.1f" % ev, "data": _data(ev), "traces": [1, 2]})
    _steps.append({"label": "%.1f" % ev, "method": "animate", "args": [["%.1f" % ev], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_d0 = _data(0.8)
_traces = [{"x": _ls, "y": [math.sqrt(l) for l in _ls], "mode": "lines", "name": "Higgs m_sigma = sqrt(lam) mu", "line": {"color": "royalblue", "width": 3}},
           dict(_d0[0], mode="lines", name="vector m_A = sqrt(2) e mu", line={"color": "firebrick", "width": 3}),
           dict(_d0[1], mode="markers", name="m_sigma = m_A at lam = 2 e^2", marker={"color": "black", "size": 10})]
_layout = {"title": "Abelian Higgs model: masses (units of mu) against lam; slide the gauge coupling e",
           "xaxis": {"title": "lam", "range": [0, 3]}, "yaxis": {"title": "mass / mu", "range": [0, 2.0]},
           "sliders": [{"active": 7, "currentvalue": {"prefix": "e = "}, "steps": _steps, "pad": {"t": 40}}], "legend": {"orientation": "h", "y": -0.25}}
_path = plotly_html("higgs_spectrum", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
_lam, _e, _mu = var("_lam _e _mu", domain="positive")
print("[PASS] m_sigma = m_A exactly when lam = 2 e^2 (sqrt(lam) mu = sqrt(2) e mu):", solve(sqrt(_lam) * _mu == sqrt(2) * _e * _mu, _lam) == [_lam == 2 * _e^2])
print("wrote figures/interactive/higgs_spectrum.html")
