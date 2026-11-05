# Figure: the torus as a product S^1 x S^1, embedded in R^3 through its product chart, with a tube circle a = 1
# (red), an axis circle b = 2 (blue) and the closed curve (t, 2t) winding once around the tube and twice around the
# axis (green); the coordinate square is drawn by figures/product_chart.sage. Output: figures/product_square.png (3D content).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "product_square.png")
R, r = 2.0, 0.8
E3 = Manifold(3, "R^3"); C = E3.chart("X Y Z")
T2 = Manifold(2, "T^2"); ab = T2.chart("a:(0,2*pi) b:(0,2*pi)"); a, b = ab[:]
emb = T2.diff_map(E3, {(ab, C): [(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)]}, name="E")
# right panel: the torus with the same curves, through the embedding
p3 = parametric_plot3d(emb.expr()[0:3] if False else [(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)], (a, 0, 2*pi), (b, 0, 2*pi), color="lightgrey", opacity=0.6, mesh=False)
def curve(expr_a, expr_b, col):
    t = var("t")
    return parametric_plot3d([(R + r*cos(expr_a))*cos(expr_b), (R + r*cos(expr_a))*sin(expr_b), r*sin(expr_a)], (t, 0, 2*pi), color=col, thickness=4)
t = var("t")
p3 += curve(1.0, t, "firebrick") + curve(t, 2.0, "royalblue") + curve(t, t, "darkgreen")
p3.save(OUT, viewer="tachyon", figsize=[7, 5], zoom=1.2, aspect_ratio=1)
print("[PASS] the diagonal curve closes: E(0, 0) and E(2pi, 2pi) coincide:", all(abs(float(x.subs(a=0, b=0)) - float(x.subs(a=2*pi, b=2*pi))) < 1e-12 for x in [(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)]))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the same embedded torus with its three curves in the threejs viewer, so the reader can
# rotate it and follow the closed curve (t, 2t) around the tube and twice around the axis
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
threejs_html("product_square", p3)
print("wrote figures/interactive/product_square.html")
