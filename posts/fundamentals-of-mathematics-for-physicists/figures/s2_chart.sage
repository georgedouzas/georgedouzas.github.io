# Figure: the spherical chart (θ, φ) on the unit sphere S², with the coordinate lines θ = const (parallels) and
# φ = const (meridians), the chart's domain 0 < θ < π, 0 < φ < 2π, and the two poles where it degenerates.
# Output: figures/s2_chart.png (a 3D scene; Sage's 3D backends emit raster images only).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_chart.png")
th, ph = var("th ph")
embed = (sin(th) * cos(ph), sin(th) * sin(ph), cos(th))          # the chart's inverse: (θ, φ) ↦ (x, y, z)
scene = parametric_plot3d(embed, (th, 0, pi), (ph, 0, 2 * pi), color="lightsteelblue", opacity=0.55, mesh=False)
for k in range(1, 6):                                            # parallels θ = kπ/6
    scene += parametric_plot3d([e.subs(th=k * pi / 6) for e in embed], (ph, 0, 2 * pi), color="black", thickness=1.5)
for k in range(12):                                              # meridians φ = kπ/6
    scene += parametric_plot3d([e.subs(ph=k * pi / 6) for e in embed], (th, 0, pi), color="gray", thickness=1)
scene += point3d((0, 0, 1), size=18, color="red") + point3d((0, 0, -1), size=18, color="red")   # the poles
p = (pi / 3, pi / 4)                                             # a sample point and its coordinate lines
scene += point3d([e.subs(th=p[0], ph=p[1]) for e in embed], size=18, color="darkgreen")
scene += parametric_plot3d([e.subs(th=p[0]) for e in embed], (ph, 0, p[1]), color="darkgreen", thickness=3)
scene += parametric_plot3d([e.subs(ph=p[1]) for e in embed], (th, 0, p[0]), color="darkgreen", thickness=3)
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.35, aspect_ratio=1)
# the quantities the text states for this figure: the metric components along the coordinate lines
g = matrix([[1, 0], [0, sin(th) ** 2]])
J = matrix([[diff(e, v) for v in (th, ph)] for e in embed])      # Jacobian of the embedding
induced = (J.transpose() * J).simplify_full()
print("[PASS] induced metric of the chart is diag(1, sin^2 theta):", bool(induced == g))
print("[PASS] chart degenerates at the poles: det g = sin^2 theta vanishes at theta = 0, pi:",
      bool(g.det().subs(th=0) == 0 and g.det().subs(th=pi) == 0))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the sphere with its parallels, meridians, poles and sample point in the threejs
# viewer, rotatable, so the degeneration of the chart at the poles can be inspected from any direction
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
threejs_html("s2_chart", scene)
print("wrote figures/interactive/s2_chart.html")
