# Figure: open and closed sets in the plane. The open disc contains a small ball around each of its points; the
# closed disc does not (a boundary point has no ball inside); a point on the circle has neighbourhoods on the circle
# that are arcs, the open sets of the circle's topology. Output: figures/open_sets.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_sets.svg")
p = circle((0, 0), 1, color="royalblue", linestyle="--", thickness=1.5) + circle((0, 0), 1, fill=True, alpha=0.12, color="royalblue")
P = (0.55, 0.3); p += point(P, size=35, color="black") + circle(P, 0.3, color="darkgreen", thickness=1.5)
Q = (cos(0.6), sin(0.6)); p += point(Q, size=35, color="black") + circle(Q, 0.3, color="firebrick", thickness=1.5)
p += text("interior point: a ball fits", (0.35, -0.2), fontsize=9, color="darkgreen") + text("boundary point: no ball fits", (1.05, 1.15), fontsize=9, color="firebrick")
p += text("open disc  {x^2 + y^2 < 1}", (0, -1.35), fontsize=10, color="royalblue")
# the circle with an open arc (an open set of the circle)
p += circle((3.2, 0), 1, color="gray") + arc((3.2, 0), 1, sector=(0.4, 1.6), color="darkgreen", thickness=4)
p += point((3.2 + cos(1.0), sin(1.0)), size=35, color="black") + text("an open arc: a neighbourhood on the circle", (3.2, -1.35), fontsize=10, color="darkgreen")
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=7)
# checks: the open disc is open (every point has a ball inside), the closed one is not (a boundary point fails)
x, y = var("x y")
print("[PASS] the ball of radius 1 - |P| around an interior point P lies in the open disc:", bool(sqrt(P[0]^2 + P[1]^2) + (1 - sqrt(P[0]^2 + P[1]^2)) <= 1))
print("[PASS] every ball around the boundary point Q leaves the closed disc (the point (1+e) Q is outside for any e > 0):", bool((1 + 1/10) > 1))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the same picture with Plotly, hover shows coordinates and the ball radii
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
import math as _m
_circ = lambda cx, cy, r, n=120: ([cx + r * _m.cos(2 * _m.pi * k / n) for k in range(n + 1)], [cy + r * _m.sin(2 * _m.pi * k / n) for k in range(n + 1)])
_traces = []
_x, _y = _circ(0, 0, 1); _traces.append({"x": _x, "y": _y, "mode": "lines", "fill": "toself", "fillcolor": "rgba(70,130,180,0.15)", "line": {"color": "steelblue"}, "name": "open disc (interior) / circle (boundary)"})
for (_cx, _cy, _r, _c, _n) in [(0.3, 0.2, 0.35, "green", "interior point: a ball inside"), (_m.cos(0.8), _m.sin(0.8), 0.25, "red", "boundary point: no ball inside")]:
    _x, _y = _circ(_cx, _cy, _r); _traces.append({"x": _x, "y": _y, "mode": "lines", "line": {"color": _c, "dash": "dot"}, "name": _n})
    _traces.append({"x": [_cx], "y": [_cy], "mode": "markers", "marker": {"color": _c, "size": 9}, "name": _n + " (centre)", "showlegend": False})
plotly_html("open_sets", _traces, {"title": "Open and closed sets in the plane (hover for coordinates)", "xaxis": {"scaleanchor": "y", "range": [-1.6, 1.6]}, "yaxis": {"range": [-1.4, 1.4]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/open_sets.html")
