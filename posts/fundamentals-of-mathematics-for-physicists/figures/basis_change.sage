# Figure: one vector, two bases. The standard grid of R^2 (grey) and the grid of the basis f1 = (2, 1), f2 = (1, 2)
# (blue); the vector v = (5, 4) has coordinates (5, 4) in the standard basis and (2, 1) in the basis f, so v =
# 2 f1 + 1 f2. Output: figures/basis_change.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basis_change.svg")
f1, f2 = vector(QQ, [2, 1]), vector(QQ, [1, 2])
v = vector(QQ, [5, 4])
p = Graphics()
for k in range(-1, 8):
    p += line([(k, -1), (k, 7)], color="lightgrey") + line([(-1, k), (7, k)], color="lightgrey")
for a in range(-2, 4):
    p += line([tuple(a * f1 - 2 * f2), tuple(a * f1 + 4 * f2)], color="royalblue", alpha=0.6)
    p += line([tuple(-2 * f1 + a * f2), tuple(4 * f1 + a * f2)], color="royalblue", alpha=0.6)
p += arrow((0, 0), tuple(f1), color="royalblue", width=2) + arrow((0, 0), tuple(f2), color="royalblue", width=2)
p += arrow((0, 0), (1, 0), color="black", width=2) + arrow((0, 0), (0, 1), color="black", width=2)
p += arrow((0, 0), tuple(v), color="firebrick", width=2.5)
p += text(r"$f_1$", (2.3, 0.8), color="royalblue", fontsize=11) + text(r"$f_2$", (0.8, 2.3), color="royalblue", fontsize=11)
p += text(r"$e_1$", (1.1, -0.3), fontsize=11) + text(r"$e_2$", (-0.35, 1.1), fontsize=11)
p += text(r"$v = 5e_1 + 4e_2 = 2f_1 + f_2$", (3.2, 4.6), color="firebrick", fontsize=11)
p.axes(False); p.set_aspect_ratio(1); p.xmin(-1); p.xmax(7); p.ymin(-1); p.ymax(7)
p.save(OUT, figsize=5)
V = QQ ^ 2
W = V.span_of_basis([f1, f2])
print("[PASS] coordinates of v in the basis (f1, f2) are (2, 1):", W.coordinates(v) == [2, 1])
print("[PASS] (f1, f2) is linearly independent:", V.linear_dependence([f1, f2]) == [])
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the two grids and the vector in Plotly; hovering a grid node shows its coordinates in
# both bases, so the reader can check v = 5 e1 + 4 e2 = 2 f1 + 1 f2 and read off the coordinates of any other node
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
_traces = []
for k in range(-1, 8):
    _traces.append({"x": [k, k], "y": [-1, 7], "mode": "lines", "line": {"color": "lightgrey", "width": 1}, "hoverinfo": "skip", "showlegend": False})
    _traces.append({"x": [-1, 7], "y": [k, k], "mode": "lines", "line": {"color": "lightgrey", "width": 1}, "hoverinfo": "skip", "showlegend": False})
for a in range(-2, 4):
    for (p0, p1) in [(a * f1 - 2 * f2, a * f1 + 4 * f2), (-2 * f1 + a * f2, 4 * f1 + a * f2)]:
        _traces.append({"x": [float(p0[0]), float(p1[0])], "y": [float(p0[1]), float(p1[1])], "mode": "lines", "line": {"color": "royalblue", "width": 1}, "opacity": 0.6, "hoverinfo": "skip", "showlegend": False})
_nodes = [(a, b) for a in range(-2, 5) for b in range(-2, 5) if -1 <= (a * f1 + b * f2)[0] <= 7 and -1 <= (a * f1 + b * f2)[1] <= 7]
_traces.append({"x": [float((a * f1 + b * f2)[0]) for a, b in _nodes], "y": [float((a * f1 + b * f2)[1]) for a, b in _nodes], "mode": "markers",
                "marker": {"color": "royalblue", "size": 7}, "name": "nodes of the f-grid",
                "text": ["standard (%d, %d); in f: (%d, %d)" % (int((a * f1 + b * f2)[0]), int((a * f1 + b * f2)[1]), a, b) for a, b in _nodes], "hoverinfo": "text"})
for (vec, col, name) in [(f1, "royalblue", "f1 = (2, 1)"), (f2, "royalblue", "f2 = (1, 2)"), (vector(QQ, [1, 0]), "black", "e1"), (vector(QQ, [0, 1]), "black", "e2"), (v, "firebrick", "v = 5 e1 + 4 e2 = 2 f1 + f2")]:
    _traces.append({"x": [0, float(vec[0])], "y": [0, float(vec[1])], "mode": "lines+markers", "line": {"color": col, "width": 3}, "marker": {"size": [0, 9], "symbol": "arrow", "angleref": "previous", "color": col}, "name": name})
plotly_html("basis_change", _traces, {"title": "One vector, two bases (hover the blue nodes)", "xaxis": {"range": [-1, 7], "scaleanchor": "y", "zeroline": False}, "yaxis": {"range": [-1, 7], "zeroline": False}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/basis_change.html")
