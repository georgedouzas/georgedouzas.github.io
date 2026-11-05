# Figure: a covector as level lines. The level lines alpha = const of the linear functional alpha(x, y) = 2x + y are
# parallel lines; the kernel is the line through the origin, and a vector is measured by how many level lines it
# crosses. Output: figures/level_sets.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "level_sets.svg")
x, y = var("x y", domain="real")
p = Graphics()
for c in range(-4, 7):
    p += implicit_plot(2 * x + y == c, (x, -2, 2), (y, -2, 2), color="royalblue" if c else "black", linewidth=1.5 if c == 0 else 0.8)
p += arrow((0, 0), (1, 1), color="firebrick", width=2) + text(r"$v,\ \alpha(v)=3$", (1.35, 1.2), color="firebrick", fontsize=10)
p += arrow((0, 0), (0.5, -1), color="darkgreen", width=2) + text(r"$w,\ \alpha(w)=0$", (1.0, -1.2), color="darkgreen", fontsize=10)
p += text(r"$\alpha = 2x + y$", (0, 2.3), fontsize=11)
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=4.5)
alpha = vector(QQ, [2, 1])
print("[PASS] alpha(v) = 3 for v = (1, 1) and alpha(w) = 0 for w = (1/2, -1):", alpha * vector(QQ, [1, 1]) == 3 and alpha * vector(QQ, [1/2, -1]) == 0)
print("wrote", os.path.relpath(OUT))
