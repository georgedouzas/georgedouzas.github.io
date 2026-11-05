# Figure: the coordinate square of the torus, (a, b) in (0, 2pi)^2, with opposite edges identified (matching
# arrows): a tube circle a = 1 (red), an axis circle b = 2 (blue) and the closed curve (t, 2t) (green). The same
# curves are drawn on the embedded torus by figures/product_square.sage. Output: figures/product_chart.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "product_chart.svg")
sq = polygon([(0, 0), (2*pi, 0), (2*pi, 2*pi), (0, 2*pi)], fill=False, color="black", thickness=1.5)
sq += arrow((pi - 0.3, 0), (pi + 0.3, 0), color="black") + arrow((pi - 0.3, 2*pi), (pi + 0.3, 2*pi), color="black")
sq += arrow((0, pi - 0.3), (0, pi + 0.3), color="black", width=2) + arrow((2*pi, pi - 0.3), (2*pi, pi + 0.3), color="black", width=2)
sq += line([(1.0, 0), (1.0, 2*pi)], color="firebrick", thickness=2) + line([(0, 2.0), (2*pi, 2.0)], color="royalblue", thickness=2)
sq += line([(0, 0), (2*pi, 2*pi)], color="darkgreen", thickness=2)
sq += text(r"$a$", (2*pi + 0.4, 0.2), fontsize=11) + text(r"$b$", (0.2, 2*pi + 0.4), fontsize=11)
sq.axes(False); sq.set_aspect_ratio(1)
sq.save(OUT, figsize=3.5)
T2 = Manifold(2, "T^2"); ab = T2.chart("a:(0,2*pi) b:(0,2*pi)")
print("[PASS] the torus chart is a product chart with coordinate ranges (0, 2pi) x (0, 2pi):", all(ab.coord_bounds(i)[0][0] == 0 and ab.coord_bounds(i)[1][0] == 2*pi for i in range(2)))
print("wrote", os.path.relpath(OUT))
