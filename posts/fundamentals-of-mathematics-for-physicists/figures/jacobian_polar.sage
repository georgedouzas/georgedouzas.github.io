# Figure: the Jacobian as an area factor. Left: two small coordinate rectangles [r0, r0 + dr] x [phi0, phi0 + dphi]
# in the (r, phi) plane, of equal area. Right: their images under the polar map (r, phi) -> (r cos phi, r sin phi),
# annular sectors whose areas are r0 times larger: the Jacobian determinant is r. Output: figures/jacobian_polar.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jacobian_polar.svg")
r, ph = var("r ph", domain="real")
F = vector([r * cos(ph), r * sin(ph)])
J = jacobian(F, (r, ph))
detJ = J.det().simplify_full()
dr, dph = 0.4, 0.5
boxes = [(0.8, 0.6, "firebrick"), (2.0, 1.9, "royalblue")]
left = Graphics()
for r0, p0, col in boxes:
    left += polygon([(r0, p0), (r0 + dr, p0), (r0 + dr, p0 + dph), (r0, p0 + dph)], color=col, alpha=0.5)
left += line([(0, 0), (3, 0)], color="black") + line([(0, 0), (0, 3)], color="black")
left += text(r"$r$", (3.1, 0), fontsize=11) + text(r"$\varphi$", (0, 3.15), fontsize=11) + text("equal areas", (1.5, 3.3), fontsize=10)
left.axes(False); left.set_aspect_ratio(1)
right = Graphics()
for r0, p0, col in boxes:
    pts = [(r0 * cos(p0 + s * dph), r0 * sin(p0 + s * dph)) for s in srange(0, 1.01, 0.05)]
    pts += [((r0 + dr) * cos(p0 + dph - s * dph), (r0 + dr) * sin(p0 + dph - s * dph)) for s in srange(0, 1.01, 0.05)]
    right += polygon(pts, color=col, alpha=0.5)
for rk in [1, 2, 3]:
    right += circle((0, 0), rk, color="lightgrey")
right += line([(0, 0), (3.2, 0)], color="black") + line([(0, 0), (0, 3.2)], color="black")
right += text(r"$x$", (3.3, 0), fontsize=11) + text(r"$y$", (0, 3.35), fontsize=11) + text(r"areas scaled by $\det J = r$", (1.6, 3.5), fontsize=10)
right.axes(False); right.set_aspect_ratio(1)
g = graphics_array([left, right])
g.save(OUT, figsize=[8, 4])
print("[PASS] the Jacobian determinant of the polar map is r:", bool(detJ == r))
a1 = (0.8 + dr) ** 2 / 2 * dph - 0.8 ** 2 / 2 * dph
a2 = (2.0 + dr) ** 2 / 2 * dph - 2.0 ** 2 / 2 * dph
print("[PASS] the ratio of the image areas equals the ratio of the midpoint radii:", abs(a2 / a1 - (2.0 + dr / 2) / (0.8 + dr / 2)) < 1e-12)
print("wrote", os.path.relpath(OUT))
