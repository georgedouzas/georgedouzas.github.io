# Figure: the gauge transformation as a coordinate change along the circle. Left: on the cylinder, the lines of
# constant y (gray) and of constant y' = y + eps(x) (red): the second coordinate system winds relative to the first.
# Right: the gauge field in the two systems, A' = A - eps'(x). SageManifolds checks, in the two-dimensional ansatz,
# that the metric dx^2 + phi^2 (dy + A dx)^2 (the Riemannian toy of Part III; the sign of the circle block plays no
# role here) written in the chart (x, y') is dx^2 + phi^2 (dy' + (A - eps') dx)^2.
# Output: figures/kk_gauge_shift.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kk_gauge_shift.svg")
M = Manifold(2, "M"); X = M.chart("x y"); x, y = X[:]
Xp = M.chart("xp yp"); xp, yp = Xp[:]
eps = function("eps"); phi = function("phi"); A = function("A")
X_to_Xp = X.transition_map(Xp, [x, y + eps(x)]); X_to_Xp.set_inverse(xp, yp - eps(xp))
g = M.metric("g"); g[0, 0] = 1 + phi(x)^2 * A(x)^2; g[0, 1] = phi(x)^2 * A(x); g[1, 1] = phi(x)^2
comp = g.comp(Xp.frame())
Ap = A(xp) - diff(eps(xp), xp)
ok = all(bool((comp[i, j, Xp].expr() - [[1 + phi(xp)^2 * Ap^2, phi(xp)^2 * Ap], [phi(xp)^2 * Ap, phi(xp)^2]][i][j]).simplify_full() == 0) for i in range(2) for j in range(i, 2))
print("metric components in the shifted chart:", comp[0, 0, Xp].expr().simplify_full(), ",", comp[0, 1, Xp].expr().simplify_full(), ",", comp[1, 1, Xp].expr().simplify_full())
print("[PASS] in the chart y' = y + eps(x) the ansatz keeps its form with A' = A - eps'(x):", ok)
xs = var("xs"); epsn = 0.8 * sin(xs); An = 0.5 + 0.2 * xs
left = Graphics()
for y0 in [k * pi / 3 for k in range(-3, 4)]:
    left += plot(y0 + 0 * xs, (xs, -3, 3), color="gray", linestyle=":")
    left += plot(y0 - epsn, (xs, -3, 3), color="firebrick", thickness=1.5)
left += text("constant $y$ (gray) and constant $y' = y + \\epsilon(x)$ (red)", (0, 4.0), fontsize=9)
left.axes_labels(["$x$", "$y$"]); left.ymin(-3.5); left.ymax(4.5)
right = plot(An, (xs, -3, 3), color="gray", thickness=2, legend_label="$A(x)$")
right += plot(An - diff(epsn, xs), (xs, -3, 3), color="firebrick", thickness=2, legend_label="$A'(x) = A - \\epsilon'(x)$")
right.axes_labels(["$x$", ""]); right.set_legend_options(loc="upper left", font_size=8)
fig = graphics_array([left, right]).matplotlib(figsize=[9, 3.8]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
