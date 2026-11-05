# Figure: the scalar potential of the Wess-Zumino model with superpotential W = m phi^2/2 + y phi^3/6,
# V = |W'(phi)|^2 = |m phi + y phi^2/2|^2 over the complex phi-plane. It is non-negative and vanishes exactly at the
# two supersymmetric vacua phi = 0 and phi = -2m/y; around phi = 0 the scalar mass squared |m|^2 equals the
# fermion mass squared. Output: figures/wz_potential.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wz_potential.svg")
m, y = 1, 1
x, yy = var("x yy"); phi = x + I * yy
W1 = m * phi + y * phi^2 / 2
V = (W1 * W1.conjugate()).expand().real_part()
print("[PASS] V = |W'|^2 vanishes at phi = 0 and at phi = -2m/y and nowhere else (solve W' = 0):",
      bool(V.subs(x=0, yy=0) == 0) and bool(V.subs(x=-2 * m / y, yy=0) == 0) and len(solve(m * var("p") + y * var("p")^2 / 2 == 0, var("p"))) == 2)
Hs = matrix([[diff(V, x, 2), diff(V, x, yy)], [diff(V, x, yy), diff(V, yy, 2)]]).subs(x=0, yy=0) / 2   # V = 1/2 m_S^2 |phi|^2 ...
print("[PASS] at phi = 0 the scalar mass matrix is |m|^2 times the identity, equal to the fermion mass squared |W''(0)|^2:",
      Hs == m^2 * identity_matrix(2))
left = contour_plot(V, (x, -3.2, 1.2), (yy, -2.2, 2.2), contours=[0.05, 0.2, 0.5, 1, 2, 4, 8], fill=False, cmap="viridis", labels=False)
left += point([(0, 0), (-2 * m / y, 0)], size=50, color="firebrick", zorder=5)
left += text("$\\phi = 0$", (0, -0.45), fontsize=9, color="firebrick"); left += text("$\\phi = -2m/y$", (-2.0, -0.45), fontsize=9, color="firebrick")
left.axes_labels(["$\\mathrm{Re}\\,\\phi$", "$\\mathrm{Im}\\,\\phi$"])
right = plot(V.subs(yy=0), (x, -3.2, 1.2), color="royalblue", thickness=2, legend_label="$V(\\phi)$, $\\phi$ real")
right += plot(m^2 * x^2, (x, -1, 1), color="firebrick", linestyle="--", legend_label="$|m|^2|\\phi|^2$")
right.axes_labels(["$\\phi$", "$V$"]); right.ymax(3); right.set_legend_options(loc="upper right", font_size=8)
fig = graphics_array([left, right]).matplotlib(figsize=[9, 4]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
gx = [-3.2 + 4.4 * i / 70 for i in range(71)]; gy = [-2.2 + 4.4 * i / 70 for i in range(71)]
def Vnum(u, v): return min(6.0, ((u + 0.5 * (u * u - v * v))**2 + (v + u * v)**2))
traces = [{"type": "surface", "x": gx, "y": gy, "z": [[Vnum(u, v) for u in gx] for v in gy], "colorscale": "Viridis",
           "cmin": 0, "cmax": 6, "showscale": False, "contours": {"z": {"show": True, "usecolormap": True, "project": {"z": True}}}},
          {"type": "scatter3d", "mode": "markers", "x": [0, -2], "y": [0, 0], "z": [0, 0], "marker": {"size": 5, "color": "red"}, "name": "supersymmetric vacua"}]
plotly_html("wz_potential", traces, {"title": "V = |m phi + y phi^2/2|^2 over the complex phi plane (m = y = 1): two zeros",
            "scene": {"xaxis": {"title": "Re phi"}, "yaxis": {"title": "Im phi"}, "zaxis": {"title": "V", "range": [0, 6]}}})
print("wrote figures/interactive/wz_potential.html")
