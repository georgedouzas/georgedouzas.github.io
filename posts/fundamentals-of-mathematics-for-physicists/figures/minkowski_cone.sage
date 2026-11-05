# Figure: the Minkowski form eta(v, v) = t^2 - x^2 on R^2 (signature (+, -)): the light cone eta = 0 and the regions
# eta > 0 (timelike) and eta < 0 (spacelike). The signature is checked with Sage's QuadraticForm. Output: figures/minkowski_cone.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minkowski_cone.svg")
x, t = var("x t", domain="real")
p = region_plot(t ** 2 - x ** 2 > 0, (x, -2, 2), (t, -2, 2), incol="lightsalmon", bordercol="white", alpha=0.6)
p += region_plot(t ** 2 - x ** 2 < 0, (x, -2, 2), (t, -2, 2), incol="lightblue", bordercol="white", alpha=0.6)
p += line([(-2, -2), (2, 2)], color="black", thickness=1.5) + line([(-2, 2), (2, -2)], color="black", thickness=1.5)
p += arrow((0, 0), (0.3, 1.2), color="firebrick", width=2) + arrow((0, 0), (1.3, 0.4), color="royalblue", width=2) + arrow((0, 0), (1.1, 1.1), color="black", width=2)
p += text(r"$\eta(v,v)>0$", (0, 1.6), fontsize=10) + text(r"$\eta(v,v)<0$", (1.4, -0.3), fontsize=10) + text(r"$\eta(v,v)=0$", (1.75, 1.45), fontsize=10)
p += text(r"$\eta = dt^2 - dx^2$", (0, 2.3), fontsize=11) + text(r"$t$", (0, -2.3), fontsize=11) + text(r"$x$", (2.3, 0), fontsize=11)
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=4.5)
eta = diagonal_matrix(QQ, [1, -1])
Q = QuadraticForm(QQ, 2 * eta)
print("[PASS] the Minkowski form on R^2 has signature (1 positive, 1 negative):", Q.signature_vector()[:2] == (1, 1))
print("[PASS] the three drawn vectors are timelike, spacelike and null:", [sign(v * eta * v) for v in [vector(QQ, [12/10, 3/10]), vector(QQ, [4/10, 13/10]), vector(QQ, [1, 1])]] == [1, -1, 0])
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the light cone t^2 - x^2 - y^2 = 0 in three dimensions with the hyperboloids
# eta(v, v) = +1 (timelike unit vectors, two sheets) and eta(v, v) = -1 (spacelike), as rotatable Plotly surfaces
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
import math as _m
_n, _mr = 40, 2.0
_th = [2 * _m.pi * k / _n for k in range(_n + 1)]
def _surf(radius, height, color, name, opacity=0.5):
    rs = [_mr * k / 24 for k in range(25)]
    return {"type": "surface", "x": [[radius(r) * _m.cos(t) for t in _th] for r in rs], "y": [[radius(r) * _m.sin(t) for t in _th] for r in rs],
            "z": [[height(r) for t in _th] for r in rs], "colorscale": [[0, color], [1, color]], "showscale": False, "opacity": opacity, "name": name, "showlegend": True, "hoverinfo": "name"}
_traces = [_surf(lambda r: r, lambda r: r, "black", "light cone eta = 0 (future)", 0.35), _surf(lambda r: r, lambda r: -r, "black", "light cone eta = 0 (past)", 0.35),
           _surf(lambda r: r, lambda r: _m.sqrt(1 + r * r), "salmon", "timelike unit vectors eta = +1 (future)"), _surf(lambda r: r, lambda r: -_m.sqrt(1 + r * r), "salmon", "timelike unit vectors eta = +1 (past)"),
           _surf(lambda r: _m.sqrt(1 + r * r), lambda r: r, "lightblue", "spacelike unit vectors eta = -1"), _surf(lambda r: _m.sqrt(1 + r * r), lambda r: -r, "lightblue", "spacelike unit vectors eta = -1", 0.5)]
plotly_html("minkowski_cone", _traces, {"title": "The Minkowski form eta = dt^2 - dx^2 - dy^2: light cone and unit hyperboloids (drag to rotate)",
            "scene": {"xaxis": {"title": "x"}, "yaxis": {"title": "y"}, "zaxis": {"title": "t"}, "aspectmode": "cube"}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/minkowski_cone.html")
