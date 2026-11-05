# Figure: the unit balls of the 3-norm and of its dual 3/2-norm on R^2, with the supporting line of the functional
# y = (1, 4)/||(1, 4)||_{3/2} touching the 3-ball at x = (1, 2)/||(1, 2)||_3. Output: figures/fa2-dual-norms.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-dual-norms.svg")
t = var("t")
def curve(p, c, lab):
    return parametric_plot((sign(cos(t)) * abs(cos(t)) ^ (2 / p), sign(sin(t)) * abs(sin(t)) ^ (2 / p)), (t, 0, 2 * pi), color=c, thickness=1.8, legend_label=lab, plot_points=800)
n3 = lambda v: (abs(v[0]) ^ 3 + abs(v[1]) ^ 3) ^ (1 / 3)
n32 = lambda v: (abs(v[0]) ^ (3 / 2) + abs(v[1]) ^ (3 / 2)) ^ (2 / 3)
x0 = vector(RDF, [1, 2]) / n3((1, 2)).n()
y0 = vector(RDF, [1, 4]) / n32((1, 4)).n()
G = curve(3, "royalblue", r"$\|x\|_3 \leq 1$") + curve(3 / 2, "darkorange", r"$\|y\|_{3/2} \leq 1$")
s = var("s")
d = vector(RDF, [-y0[1], y0[0]])
G += parametric_plot(tuple(x0 + s * d), (s, -1.2, 1.2), color="gray", linestyle="dashed", legend_label=r"$\varphi = 1$")
G += point(tuple(x0), color="royalblue", size=50) + point(tuple(y0), color="darkorange", size=50)
G += text(r"$x$", (x0[0] + 0.12, x0[1] + 0.05), color="royalblue") + text(r"$y$", (y0[0] + 0.12, y0[1]), color="darkorange")
G.set_legend_options(loc="lower left", font_size=9)
G.save(OUT, figsize=[5, 5], aspect_ratio=1, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5)
print("[PASS] ||x||_3 = 1, ||y||_{3/2} = 1 and <y, x> = 1:", abs(n3(x0) - 1) < 1e-12 and abs(n32(y0) - 1) < 1e-12 and abs(x0.dot_product(y0) - 1) < 1e-12)
samples = [vector(RDF, [cos(2 * pi * k / 4000), sin(2 * pi * k / 4000)]) for k in range(4000)]
mx = max(abs(y0.dot_product(v / n3(v))) for v in samples)
print("[PASS] sup over the 3-sphere of <y, x> is 1 = ||y||_{3/2} (sampled, to 1e-6):", abs(mx - 1) < 1e-6)
print("[PASS] ||(1, 4)||_{3/2} = 9^(2/3) and the maximiser of x1 + 4 x2 on the 3-sphere is (1, 2)/9^(1/3):", bool(abs((n32((1, 4)) - 9 ^ (2 / 3)).n()) < 1e-12) and bool(abs((n3((1, 2)) - 9 ^ (1 / 3)).n()) < 1e-12))
print("wrote", os.path.relpath(OUT))
