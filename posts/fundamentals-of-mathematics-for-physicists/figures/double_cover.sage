# Figure: the double cover SU(2) -> SO(3) on a one-parameter subgroup. Left: the circle of elements exp(t T_3) in
# SU(2), t in [0, 4 pi), which returns to the identity only after t = 4 pi (exp(2 pi T_3) = -1, marked). Right:
# the circle of rotations R(t) about the Z-axis in SO(3), of period 2 pi. The dotted lines show the two-to-one map
# g -> Ad_g: t and t + 2 pi give the same rotation. Computed with Sage's matrix exponential.
# Output: figures/double_cover.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "double_cover.svg")
t = var("t", domain="real")
T3 = matrix(SR, [[-I/2, 0], [0, I/2]])
g = (t * T3).exp().simplify_full()
L3 = matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])
R = (t * L3).exp().apply_map(lambda e: e.rectform().simplify_full())
p = Graphics()
p += circle((0, 0), 1, color="black", thickness=1.2) + circle((3.2, 0), 1, color="black", thickness=1.2)
p += text(r"$e^{tT_3}\in SU(2)$, $t\in[0,4\pi)$", (0, 1.4), fontsize=10) + text(r"$R(t)\in SO(3)$, $t\in[0,2\pi)$", (3.2, 1.4), fontsize=10)
for k, tk in enumerate([0, pi/2, pi, 3*pi/2, 2*pi, 5*pi/2, 3*pi, 7*pi/2]):
    a = float(tk) / 2                                  # position on the SU(2) circle: one turn = 4 pi
    b = float(tk)                                      # position on the SO(3) circle: one turn = 2 pi
    P = (cos(a), sin(a)); Q = (3.2 + cos(b), sin(b))
    col = "firebrick" if tk < 2*pi else "royalblue"
    p += point(P, color=col, size=35) + line([P, Q], color=col, linestyle="dotted", alpha=0.6)
    p += text("$t=%s$" % latex(tk), (1.25 * P[0], 1.25 * P[1]), fontsize=7, color=col)
p += point((3.2 + 1, 0), color="black", size=50) + text("$1$", (4.5, 0), fontsize=9)
p += text(r"$e^{2\pi T_3}=-1$", (-1.0, -0.3), fontsize=8, color="royalblue") + text(r"$1$", (1.0, -0.25), fontsize=9)
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=[8, 3.2])
print("[PASS] exp(t T_3) = diag(e^{-it/2}, e^{it/2}) and exp(2 pi T_3) = -1:", bool(g == matrix(SR, [[exp(-I*t/2), 0], [0, exp(I*t/2)]])) and bool(g.subs(t=2*pi) == -identity_matrix(SR, 2)))
print("[PASS] exp(t L_3) is the rotation by t about Z, with period 2 pi:", bool(R == matrix(SR, [[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])) and bool(R.subs(t=2*pi) == identity_matrix(SR, 3)))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the one-parameter subgroups e^{t n.L} of SO(3) acting on a vector v of the unit
# sphere, for three axes n, in Plotly 3D. Hovering a marker shows t, the rotated vector, and the two elements
# +-e^{t n.T} of SU(2) covering the rotation (the second is e^{(t + 2 pi) n.T}), checked numerically through Ad.
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
_L = [matrix(RDF, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(RDF, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(RDF, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
_s = [matrix(CDF, [[0, 1], [1, 0]]), matrix(CDF, [[0, -1j], [1j, 0]]), matrix(CDF, [[1, 0], [0, -1]])]
_T = [-0.5j * sa for sa in _s]
def _rot(n, u):
    N = sum(n[a] * _L[a] for a in range(3)); return identity_matrix(RDF, 3) + sin(u) * N + (1 - cos(u)) * N * N
def _lift(n, u):
    return cos(u / 2) * identity_matrix(CDF, 2) - 1j * sin(u / 2) * sum(n[a] * _s[a] for a in range(3))
def _Ad(g):
    gi = g.inverse(); return matrix(RDF, 3, 3, lambda a, b: (-2 * (_T[a] * g * _T[b] * gi).trace()).real())
_v = vector(RDF, [0.8, 0.0, 0.6])
_axes = [(vector(RDF, [0, 0, 1]), "firebrick", "n = Z"), (vector(RDF, [1, 0, 0]), "royalblue", "n = X"), (vector(RDF, [1, 1, 1]) / sqrt(3.0), "darkgreen", "n = (1,1,1)/sqrt 3")]
_traces, _err = [], 0.0
_U = [k * pi / 20 for k in range(41)]; _V = [k * pi / 20 for k in range(21)]
_traces.append({"type": "surface", "x": [[float(sin(b) * cos(a)) for a in _U] for b in _V], "y": [[float(sin(b) * sin(a)) for a in _U] for b in _V],
                "z": [[float(cos(b)) for a in _U] for b in _V], "opacity": 0.15, "showscale": False, "colorscale": [[0, "lightgrey"], [1, "lightgrey"]], "hoverinfo": "skip"})
for n, col, name in _axes:
    ts = [k * 2 * pi / 120 for k in range(121)]
    pts = [_rot(n, float(t)) * _v for t in ts]
    _traces.append({"type": "scatter3d", "mode": "lines", "x": [p[0] for p in pts], "y": [p[1] for p in pts], "z": [p[2] for p in pts], "line": {"color": col, "width": 5}, "name": "orbit of v, " + name})
    _traces.append({"type": "scatter3d", "mode": "lines", "x": [-1.3 * n[0], 1.3 * n[0]], "y": [-1.3 * n[1], 1.3 * n[1]], "z": [-1.3 * n[2], 1.3 * n[2]], "line": {"color": col, "width": 2, "dash": "dash"}, "showlegend": False, "hoverinfo": "skip"})
    mk, txt = [], []
    for k in range(12):
        t = k * pi / 6
        R = _rot(n, float(t)); g = _lift(n, float(t))
        _err = max(_err, (_Ad(g) - R).norm(), (_Ad(-g) - R).norm())
        p = R * _v; mk.append(p)
        txt.append("t = %d pi/6<br>R(t) v = (%.3f, %.3f, %.3f)<br>covered by +-g, g = [[%.3f%+.3fi, %.3f%+.3fi], [%.3f%+.3fi, %.3f%+.3fi]]" % (
            k, p[0], p[1], p[2], g[0, 0].real(), g[0, 0].imag(), g[0, 1].real(), g[0, 1].imag(), g[1, 0].real(), g[1, 0].imag(), g[1, 1].real(), g[1, 1].imag()))
    _traces.append({"type": "scatter3d", "mode": "markers", "x": [p[0] for p in mk], "y": [p[1] for p in mk], "z": [p[2] for p in mk], "marker": {"color": col, "size": 4}, "text": txt, "hoverinfo": "text", "showlegend": False})
_traces.append({"type": "scatter3d", "mode": "markers+text", "x": [_v[0]], "y": [_v[1]], "z": [_v[2]], "marker": {"color": "black", "size": 6}, "text": ["v"], "name": "v = (0.8, 0, 0.6)"})
plotly_html("double_cover", _traces, {"title": "One-parameter subgroups exp(t n.L) of SO(3) acting on v (hover the markers)",
            "scene": {"aspectmode": "cube", "xaxis": {"range": [-1.3, 1.3]}, "yaxis": {"range": [-1.3, 1.3]}, "zaxis": {"range": [-1.3, 1.3]}}, "legend": {"orientation": "h"}})
print("[PASS] Ad_{+-exp(t n.T)} = exp(t n.L) at the 36 sampled points (max error < 1e-12):", _err < 1e-12)
print("wrote figures/interactive/double_cover.html")
