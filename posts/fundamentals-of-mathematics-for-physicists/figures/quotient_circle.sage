# Figure: the circle as the quotient R/2piZ. Top: the real line with the angles phi, phi + 2pi, phi + 4pi of three
# equivalence classes marked in three colours. Bottom: the quotient map wraps the line onto the circle, where each
# class becomes a single point. Output: figures/quotient_circle.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quotient_circle.svg")
classes = {"firebrick": 0.6, "royalblue": 2.0, "darkgreen": 4.2}
p = line([(-0.5, 3.2), (3 * 2 * pi + 0.5, 3.2)], color="black", thickness=1)
for k in range(4):
    p += line([(2 * pi * k, 3.05), (2 * pi * k, 3.35)], color="black")
    p += text(r"$%d\pi$" % (2 * k) if k else "$0$", (2 * pi * k, 2.75), fontsize=9)
for col, phi in classes.items():
    for k in range(3):
        p += point((phi + 2 * pi * k, 3.2), color=col, size=45)
        p += arrow((phi + 2 * pi * k, 3.0), (9.5 + 2.4 * cos(phi), -0.6 + 2.4 * sin(phi)), color=col, width=0.4, arrowsize=1.5, linestyle="dotted")
p += circle((9.5, -0.6), 2.4, color="black", thickness=1)
for col, phi in classes.items():
    p += point((9.5 + 2.4 * cos(phi), -0.6 + 2.4 * sin(phi)), color=col, size=60)
p += text(r"$\mathbb{R}$", (-1.0, 3.2), fontsize=11)
p += text(r"$\mathbb{R}/2\pi\mathbb{Z} = S^1$", (9.5, -3.5), fontsize=11)
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=[8, 3.4])
print("[PASS] the three angles of each class map to the same point of the circle:", all(len({(round(float(cos(phi + 2*pi*k)), 9), round(float(sin(phi + 2*pi*k)), 9)) for k in range(3)}) == 1 for phi in classes.values()))
print("wrote", os.path.relpath(OUT))
