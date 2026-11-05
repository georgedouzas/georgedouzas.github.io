# Figure: the non-abelian Kaluza-Klein ansatz on M4 x S^2. Along a spacetime direction x the sphere fibres are
# "tilted" by the vector field xi(x) = A^a_mu(x) K_a on each fibre, drawn as arrows on a row of spheres for a gauge
# field with A^1 growing and A^3 oscillating with x. The three Killing vectors are those of Part I, with brackets
# [K_a, K_b] = -eps_abc K_c and inner products K_a . K_b = delta_ab - n_a n_b (checked symbolically).
# Output: figures/s2_gauge_fields.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_gauge_fields.svg")
th, ph = var("th ph")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
X3 = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
def bracket(U, V): return vector([sum(U[n] * diff(V[m], [th, ph][n]) - V[n] * diff(U[m], [th, ph][n]) for n in range(2)) for m in range(2)])
print("[PASS] [K_a, K_b] = -eps_abc K_c for the three Killing vectors of the sphere:",
      all((bracket(K[a], K[b]) + sum(lc(a, b, c) * K[c] for c in range(3))).apply_map(lambda e: e.simplify_full()).is_zero() for a in range(3) for b in range(3)))
h = diagonal_matrix([1, sin(th)^2])
G = matrix(SR, 3, 3, lambda a, b: (K[a] * h * K[b]).simplify_full())
print("[PASS] K_a . K_b = delta_ab - n_a n_b pointwise on the unit sphere:", (G - (identity_matrix(3) - X3.column() * X3.row())).apply_map(lambda e: e.simplify_full()).is_zero())
# the picture: spheres at x = -2, -1, 0, 1, 2 with the tangent field xi = A^1(x) K_1 + A^3(x) K_3 (orthographic view from +y)
def xi_tangent(xv, t0, p0):
    A1, A3 = 0.5 * (xv + 2) / 4, 0.6 * sin(xv)
    v = A1 * K[0] + A3 * K[2]
    v = v.subs(th=t0, ph=p0)
    # push forward to R^3: dX/dth * v^th + dX/dph * v^ph
    return (diff(X3, th) * v[0] + diff(X3, ph) * v[1]).subs(th=t0, ph=p0)
p = Graphics()
for i, xv in enumerate([-2, -1, 0, 1]):
    cx = 2.6 * i
    p += circle((cx, 0), 1, color="steelblue")
    for t0 in [k * pi / 6 for k in range(1, 6)]:
        for p0 in [k * pi / 4 for k in range(8)]:
            pt = X3.subs(th=t0, ph=p0)
            if pt[1] < 0: continue                                       # back side hidden
            v = xi_tangent(xv, t0, p0) * 0.45
            p += arrow((cx + pt[0], pt[2]), (cx + pt[0] + v[0], pt[2] + v[2]), color="firebrick", width=0.8, arrowsize=1.5)
    p += text("$x = %d$" % xv, (cx, -1.35), fontsize=9)
p += text(r"fibre $S^2$ over each $x$, with the tilt $\xi = A^a_\mu(x)\,K_a$ (red)", (3.9, 1.5), fontsize=10)
p.axes(False); p.set_aspect_ratio(1); p.xmin(-1.3); p.xmax(9.1)
p.save(OUT, figsize=[10, 3.2])
print("wrote", os.path.relpath(OUT))
# the Killing vectors seen in R^3: the push-forward of K_a by the embedding is the rotation field e_a x n
Jac = matrix(SR, 3, 2, lambda i, m: diff(X3[i], [th, ph][m]))
E3 = identity_matrix(SR, 3).columns()
print("[PASS] the push-forward of K_a to R^3 is the rotation field e_a x n:", all((Jac * K[a] - E3[a].cross_product(X3)).simplify_full().is_zero() for a in range(3)))
# interactive version: the three Killing vector fields on the sphere in 3D (cones), toggled from the legend
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
_N = 160; _gold = float(pi * (3 - sqrt(5)))
_pts = []
for _i in range(_N):
    _z = 1 - 2 * (_i + 0.5) / _N; _r = sqrt(1 - _z^2)
    _pts.append((float(_r * cos(_gold * _i)), float(_r * sin(_gold * _i)), float(_z)))
_traces = []
_u = [k * 2 * float(pi) / 40 for k in range(41)]; _v = [k * float(pi) / 20 for k in range(21)]
_traces.append({"type": "surface", "x": [[sin(b) * cos(a) for a in _u] for b in _v], "y": [[sin(b) * sin(a) for a in _u] for b in _v],
                "z": [[cos(b) for a in _u] for b in _v], "opacity": 0.35, "showscale": False, "colorscale": [[0, "#dbe4f0"], [1, "#dbe4f0"]], "hoverinfo": "skip", "name": "S^2"})
for _a, _col in zip(range(3), ["#c0392b", "#2471a3", "#1e8449"]):
    _vec = [E3[_a].cross_product(vector(p)) for p in _pts]           # = push-forward of K_a (checked above)
    _traces.append({"type": "cone", "x": [p[0] for p in _pts], "y": [p[1] for p in _pts], "z": [p[2] for p in _pts],
                    "u": [w[0] for w in _vec], "v": [w[1] for w in _vec], "w": [w[2] for w in _vec], "sizemode": "absolute", "sizeref": 0.35,
                    "anchor": "tail", "colorscale": [[0, _col], [1, _col]], "showscale": False, "name": "K_%d (rotation about the %s-axis)" % (_a + 1, "xyz"[_a]),
                    "showlegend": True, "visible": True if _a == 2 else "legendonly"})
plotly_html("s2_gauge_fields", _traces, {"title": "Killing vectors K_a = e_a x n of the sphere (click the legend to switch K_1, K_2, K_3)",
            "scene": {"aspectmode": "data", "xaxis": {"title": "x"}, "yaxis": {"title": "y"}, "zaxis": {"title": "z"}}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/s2_gauge_fields.html")
