# Figure: Cayley graphs of two groups of order six, drawn with Sage's group and graph modules. Left: the cyclic
# group Z_6 with generator 1, a single six-cycle. Right: the symmetric group S_3 with generators (1 2) (red edges)
# and (1 2 3) (blue edges): two triangles joined by three red edges, the shape of a non-abelian group.
# Output: figures/cayley_graph.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cayley_graph.svg")
Z6 = CyclicPermutationGroup(6)
S3 = SymmetricGroup(3)
s, c = S3((1, 2)), S3((1, 2, 3))
G1 = Z6.cayley_graph(generators=[Z6.gen()])
G2 = S3.cayley_graph(generators=[s, c])
pos1 = {v: (cos(pi / 3 * i), sin(pi / 3 * i)) for i, v in enumerate(Z6.list())}
pos2 = {}
for i, e in enumerate([S3.one(), c, c * c]):
    ang = 2 * pi * i / 3 + pi / 2
    pos2[e] = (cos(ang), sin(ang))
    pos2[e * s] = (0.5 * cos(ang), 0.5 * sin(ang))          # the red edge e -> e*s is radial
edge_colors = {"firebrick": [(u, v, l) for (u, v, l) in G2.edges() if l == s], "royalblue": [(u, v, l) for (u, v, l) in G2.edges() if l == c]}
p1 = G1.plot(pos=pos1, vertex_labels=False, vertex_size=220, vertex_colors={"lightgrey": Z6.list()}, edge_colors={"royalblue": list(G1.edges())}, edge_thickness=1.5)
for i, v in enumerate(Z6.list()):
    p1 += text(str(i), (1.25 * cos(pi / 3 * i), 1.25 * sin(pi / 3 * i)), fontsize=10)
p1 += text(r"$\mathbb{Z}_6$, generator $1$", (0, 1.6), fontsize=10)
p2 = G2.plot(pos=pos2, vertex_labels=False, vertex_size=220, vertex_colors={"lightgrey": list(S3)}, edge_colors=edge_colors, edge_thickness=1.5)
for e, (x, y) in pos2.items():
    r = float(sqrt(x**2 + y**2))
    f = 1.28 if r > 0.8 else 0.45
    p2 += text("e" if e == S3.one() else str(e), (x * f, y * f), fontsize=8)
p2 += text(r"$S_3$, generators $(12)$ (red) and $(123)$ (blue)", (0, 1.6), fontsize=10)
for p in (p1, p2):
    p.axes(False); p.set_aspect_ratio(1)
g = graphics_array([p1, p2]); fig = g.matplotlib(figsize=[8, 4.2], axes=False); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] Z_6 is abelian and S_3 is not:", Z6.is_abelian() and not S3.is_abelian())
print("[PASS] (12)(13) and (13)(12) differ, and (123) has order 3:", S3((1, 2)) * S3((1, 3)) != S3((1, 3)) * S3((1, 2)) and c.order() == 3)
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the multiplication table of S_3 as a Plotly heatmap. Rows g and columns h are
# ordered with A_3 = {e, (123), (132)} first; cell (g, h) shows the product g h in the right-to-left convention of the
# text (first h, then g; Sage's h*g), and its colour is the sign of g h, so the table splits into four 3 x 3 blocks:
# the quotient S_3/A_3 = Z_2. Hovering a cell shows the product.
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
_order = [S3.one(), S3((1, 2, 3)), S3((1, 3, 2)), S3((1, 2)), S3((1, 3)), S3((2, 3))]
_name = lambda g: "e" if g == S3.one() else str(g).replace(",", " ")
_prod = lambda g, h: h * g                       # the text's product g h
_z = [[int(_prod(g, h).sign()) for h in _order] for g in _order]
_txt = [["%s · %s = %s" % (_name(g), _name(h), _name(_prod(g, h))) for h in _order] for g in _order]
_labels = [_name(g) for g in _order]
_traces = [{"type": "heatmap", "z": _z, "x": _labels, "y": _labels, "text": _txt, "hoverinfo": "text",
            "colorscale": [[0, "#f4a582"], [1, "#92c5de"]], "showscale": False, "xgap": 2, "ygap": 2}]
_ann = [{"x": _labels[j], "y": _labels[i], "text": _name(_prod(g, h)), "showarrow": False, "font": {"size": 14}}
        for i, g in enumerate(_order) for j, h in enumerate(_order)]
plotly_html("cayley_graph", _traces, {"title": "Multiplication table of S_3: row g, column h, entry g h (blue: even, orange: odd)",
            "xaxis": {"title": "h (applied first)", "side": "top"}, "yaxis": {"title": "g", "autorange": "reversed"}, "annotations": _ann})
_blocks = all(_prod(g, h).sign() == g.sign() * h.sign() for g in _order for h in _order)
print("[PASS] the sign of g h is the product of the signs (the table has the block pattern of S_3/A_3 = Z_2):", _blocks)
print("[PASS] every row and column of the table is a permutation of S_3:", all(len({_prod(g, h) for h in _order}) == 6 and len({_prod(h, g) for h in _order}) == 6 for g in _order))
print("wrote figures/interactive/cayley_graph.html")
