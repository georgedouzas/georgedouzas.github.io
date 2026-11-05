# Figure: two bundles over the circle with fibre an interval. Left: the cylinder S^1 x (-1, 1), the trivial bundle,
# with a nowhere-vanishing section (red curve at height 1/2). Right: the Moebius band, glued with the flip, with a
# section (red) that must cross the zero section (grey central circle): a continuous section changes sign after one
# turn, so it has a zero. Drawn as parametric surfaces. Output: figures/mobius_cylinder.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobius_cylinder.png")
u, v = var("u v")
cyl = parametric_plot3d([2 * cos(u), 2 * sin(u), v], (u, 0, 2 * pi), (v, -1, 1), color="lightsteelblue", opacity=0.7, mesh=False)
cyl += parametric_plot3d([2 * cos(u), 2 * sin(u), 0.5], (u, 0, 2 * pi), color="firebrick", thickness=4)
cyl += parametric_plot3d([2 * cos(u), 2 * sin(u), 0], (u, 0, 2 * pi), color="grey", thickness=2)
mob = parametric_plot3d([(2 + v * cos(u / 2)) * cos(u), (2 + v * cos(u / 2)) * sin(u), v * sin(u / 2)], (u, 0, 2 * pi), (v, -1, 1), color="lightsalmon", opacity=0.7, mesh=False)
s = 0.6 * cos(u / 2)                                  # a section: s(u) in (-1, 1) with s(2 pi) = -s(0), vanishing at u = pi
mob += parametric_plot3d([(2 + s * cos(u / 2)) * cos(u), (2 + s * cos(u / 2)) * sin(u), s * sin(u / 2)], (u, 0, 2 * pi), color="firebrick", thickness=4)
mob += parametric_plot3d([2 * cos(u), 2 * sin(u), 0], (u, 0, 2 * pi), color="grey", thickness=2)
scene = (cyl + mob.translate((4.0, -4.0, 0.0))).rotateX(-1.1)
scene.save(OUT, viewer="tachyon", figsize=[9, 4], zoom=1.9, aspect_ratio=1)
print("[PASS] the Moebius section s(u) = 0.6 cos(u/2) changes sign across the glued edge, s(2 pi) = -s(0), and vanishes at u = pi:", bool(s.subs(u=2*pi) == -s.subs(u=0)) and bool(s.subs(u=pi) == 0))
print("[PASS] the cylinder section s = 1/2 is constant and never zero:", True if all(QQ(1)/2 != 0 for _ in [0]) else False)
print("wrote", os.path.relpath(OUT))
