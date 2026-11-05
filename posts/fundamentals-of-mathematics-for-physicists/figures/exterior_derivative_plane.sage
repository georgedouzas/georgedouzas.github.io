# Figure: one-forms and their exterior derivatives on the plane, with SageManifolds. Left: the rotation form
# omega = -y dx + x dy, drawn through the vector field with the same components (arrows); its exterior derivative
# d omega = 2 dx ^ dy is a constant two-form, shaded. Right: the radial form eta = x dx + y dy = d(r^2/2) is exact,
# d eta = 0, and the arrows are a gradient field. Output: figures/exterior_derivative_plane.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exterior_derivative_plane.svg")
M = Manifold(2, "R^2"); X = M.chart("x y"); x, y = X[:]
om = M.one_form(-y, x, name="omega"); eta = M.one_form(x, y, name="eta")
dom = om.exterior_derivative(); deta = eta.exterior_derivative()
f = M.scalar_field((x**2 + y**2) / 2, name="f")
xx, yy = var("xx yy", domain="real")
left = region_plot(xx**2 + yy**2 < 9, (xx, -2, 2), (yy, -2, 2), incol="lightsalmon", alpha=0.35, bordercol="white")
left += plot_vector_field((-yy, xx), (xx, -2, 2), (yy, -2, 2), color="firebrick", plot_points=9)
left += text(r"$\omega=-y\,dx+x\,dy$, $d\omega=2\,dx\wedge dy$ (shaded, constant)", (0, 2.4), fontsize=9)
right = plot_vector_field((xx, yy), (xx, -2, 2), (yy, -2, 2), color="royalblue", plot_points=9)
for rk in [0.5, 1.0, 1.5, 2.0]:
    right += circle((0, 0), rk, color="lightgrey")
right += text(r"$\eta=x\,dx+y\,dy=d(r^2/2)$, $d\eta=0$ (level sets of $r^2/2$ in grey)", (0, 2.4), fontsize=9)
for p in (left, right):
    p.axes(False); p.set_aspect_ratio(1)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9, 4.4], axes=False); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] d(-y dx + x dy) = 2 dx^dy and d(x dx + y dy) = 0:", bool(dom[0, 1].expr() == 2) and deta == 0)
print("[PASS] x dx + y dy is exact, the differential of r^2/2, and d(d f) = 0:", f.differential() == eta and f.differential().exterior_derivative() == 0)
print("wrote", os.path.relpath(OUT))
