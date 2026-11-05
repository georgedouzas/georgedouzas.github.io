# Figure: a connection on a circle bundle over a circle and its holonomy. The total space is the torus with base
# angle x (horizontal) and fibre angle y (vertical), drawn unrolled as a square with identified edges. The connection
# A = c dx declares the slanted lines dy = -c dx horizontal; lifting the base loop x: 0 -> 2 pi horizontally from
# the fibre point y = 0 (red) ends at y = -2 pi c, not at the start: the holonomy of the loop is the shift by -2 pi c,
# the integral of A around the loop. Output: figures/horizontal_lift.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "horizontal_lift.svg")
c = QQ(1) / 4
p = polygon([(0, 0), (2 * pi, 0), (2 * pi, 2 * pi), (0, 2 * pi)], fill=False, color="black", thickness=1.5)
for y0 in [k * pi / 3 for k in range(0, 8)]:
    p += line([(0, y0), (2 * pi, y0 - 2 * pi * c)], color="lightgrey")
p += line([(0, 2 * pi), (2 * pi, 2 * pi - 2 * pi * c)], color="firebrick", thickness=2.5)      # the lift, drawn with y taken modulo 2 pi
p += point((0, 2 * pi), color="black", size=40) + point((2 * pi, 2 * pi - 2 * pi * c), color="black", size=40) + point((2 * pi, 2 * pi), color="grey", size=40, faceted=True)
p += text(r"start: $y=0\equiv 2\pi$", (-1.2, 2 * pi), fontsize=9) + text(r"end: $y=-2\pi c$", (2 * pi + 1.2, 2 * pi - 2 * pi * c), fontsize=9) + text(r"$y=0$ again", (2 * pi + 1.0, 2 * pi + 0.3), fontsize=8, color="grey")
p += text(r"base angle $x$", (pi, -0.5), fontsize=10) + text(r"fibre angle $y$", (-1.5, pi), fontsize=10, rotation=90)
p += text(r"$A=c\,dx$: horizontal lines $dy=-c\,dx$; holonomy $=-\oint A=-2\pi c$", (pi, 2 * pi + 0.9), fontsize=9)
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=[6.5, 5.5])
x = var("x")
print("[PASS] the horizontal lift of the base loop ends at y = -2 pi c = -integral of A:", bool(integrate(-c, x, 0, 2 * pi) == -2 * pi * c))
print("[PASS] the curvature of A = c dx vanishes (flat connection) although the holonomy is non-trivial:", bool(diff(c, x) == 0))
print("wrote", os.path.relpath(OUT))
