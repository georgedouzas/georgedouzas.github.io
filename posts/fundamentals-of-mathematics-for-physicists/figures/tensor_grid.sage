# Figure: tensors as arrays. The components of the elementary tensor u (x) v are the products u^i v^j, an outer
# product of rank one (left); the sum of two elementary tensors has rank two (middle); a general element of
# V (x) V with all nine components independent (right). Each cell is coloured by its value.
# Output: figures/tensor_grid.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensor_grid.svg")
u, v = vector(QQ, [1, 2, 3]), vector(QQ, [1, 0, -1])
w, z = vector(QQ, [0, 1, 1]), vector(QQ, [2, 1, 0])
T1 = u.outer_product(v)
T2 = T1 + w.outer_product(z)
T3 = matrix(QQ, 3, 3, [1, 2, 0, -1, 3, 1, 0, 2, -2])
def shade(val):
    s = float(val) / 6.0
    return (1 - 0.6 * max(-s, 0), 1 - 0.6 * abs(s), 1 - 0.6 * max(s, 0)) if s < 0 else (1 - 0.6 * s, 1 - 0.6 * s, 1)
def panel(M, title, x0):
    p = Graphics()
    for i in range(3):
        for j in range(3):
            x, y = x0 + j, 2 - i
            p += polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)], color=shade(M[i, j]), edgecolor="grey")
            p += text(str(M[i, j]), (x + 0.5, y + 0.5), color="black", fontsize=11)
    p += text(title, (x0 + 1.5, 3.45), fontsize=11, color="black")
    for i in range(3):
        p += text("$i=%d$" % (i + 1), (x0 - 0.45, 2.5 - i), fontsize=8, color="grey")
        p += text("$j=%d$" % (i + 1), (x0 + 0.5 + i, -0.3), fontsize=8, color="grey")
    return p
p = panel(T1, r"$u\otimes v$: rank 1", 0) + panel(T2, r"$u\otimes v + w\otimes z$: rank 2", 4.5) + panel(T3, "general: rank 3", 9)
fig = p.matplotlib(figsize=[10, 3.4], axes=False, aspect_ratio=1, xmin=-0.8, xmax=12.2, ymin=-0.6, ymax=3.8)
fig.savefig(OUT, bbox_inches="tight")
print("[PASS] (u (x) v)^{ij} = u^i v^j has rank 1, the sum of two elementary tensors rank 2, the general tensor rank 3:", T1.rank() == 1 and T2.rank() == 2 and T3.rank() == 3)
print("[PASS] V (x) V has dimension 9 = 3 * 3:", matrix(QQ, 3, 3).parent().dimension() == 9)
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the three component arrays as Plotly heatmaps; hovering a cell shows (i, j) and the
# component, and for the elementary tensor the factorisation u^i v^j
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
def _heat(M, x0, name, texts):
    return {"type": "heatmap", "x": [x0 + j for j in range(3)], "y": [3 - i for i in range(3)], "z": [[int(M[i, j]) for j in range(3)] for i in range(3)],
            "text": [[texts(i, j) for j in range(3)] for i in range(3)], "hoverinfo": "text", "colorscale": "RdBu", "zmid": 0, "zmin": -6, "zmax": 6, "showscale": False, "name": name, "xgap": 2, "ygap": 2}
_traces = [_heat(T1, 0, "u (x) v", lambda i, j: "(u (x) v)^{%d%d} = u^%d v^%d = %d * %d = %d" % (i + 1, j + 1, i + 1, j + 1, u[i], v[j], T1[i, j])),
           _heat(T2, 4, "u (x) v + w (x) z", lambda i, j: "(u (x) v + w (x) z)^{%d%d} = %d*%d + %d*%d = %d" % (i + 1, j + 1, u[i], v[j], w[i], z[j], T2[i, j])),
           _heat(T3, 8, "general", lambda i, j: "T^{%d%d} = %d (independent component)" % (i + 1, j + 1, T3[i, j]))]
_ann = [{"x": x, "y": 3.9, "text": t, "showarrow": False, "font": {"size": 13}} for x, t in [(1, "u (x) v: rank 1"), (5, "u (x) v + w (x) z: rank 2"), (9, "general: rank 3")]]
_ann += [{"x": x0 + j, "y": 3 - i, "text": str(M[i, j]), "showarrow": False, "font": {"color": "black"}} for M, x0 in [(T1, 0), (T2, 4), (T3, 8)] for i in range(3) for j in range(3)]
plotly_html("tensor_grid", _traces, {"title": "Tensors as arrays (hover a cell for its component)", "xaxis": {"visible": False, "range": [-0.7, 10.7]}, "yaxis": {"visible": False, "range": [0.3, 4.3], "scaleanchor": "x"}, "annotations": _ann, "showlegend": False})
print("wrote figures/interactive/tensor_grid.html")
