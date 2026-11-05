# Figure: unit balls of the p-norms on R^2 for p = 1, 3/2, 2, 4 and infinity. The [PASS] lines check the parallelogram
# law (it holds only for p = 2) and the constants between the norms. Output: figures/fa2-unit-balls.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-unit-balls.svg")
t = var("t")
G = Graphics()
cols = {1: "firebrick", 1.5: "darkorange", 2: "royalblue", 4: "darkgreen"}
for p, c in cols.items():
    x = sign(cos(t)) * abs(cos(t)) ^ (2 / p)
    y = sign(sin(t)) * abs(sin(t)) ^ (2 / p)
    G += parametric_plot((x, y), (t, 0, 2 * pi), color=c, thickness=1.8, legend_label=r"$p = %s$" % (p if p != 1.5 else "3/2"), plot_points=800)
G += polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)], fill=False, color="purple", thickness=1.8, legend_label=r"$p = \infty$")
G.set_legend_options(loc="upper right", font_size=9)
G.save(OUT, figsize=[5, 5], aspect_ratio=1, xmin=-1.3, xmax=1.3, ymin=-1.3, ymax=1.3)
def pn(v, p):
    return max(abs(a) for a in v) if p == oo else sum(abs(a) ^ p for a in v) ^ (1 / p)
def D(u, v, p):
    s = vector(u) + vector(v); d = vector(u) - vector(v)
    return pn(s, p) ^ 2 + pn(d, p) ^ 2 - 2 * pn(u, p) ^ 2 - 2 * pn(v, p) ^ 2
print("[PASS] the parallelogram law holds for p = 2 at (1,0),(0,1) and (1,1),(1,-1):", bool(D((1, 0), (0, 1), 2) == 0) and bool(D((1, 1), (1, -1), 2) == 0))
print("[PASS] it fails for p = 1, 3/2, 4, infinity:", all(bool(D((1, 0), (0, 1), p) != 0) for p in [1, 3 / 2, 4, oo]))
print("[PASS] D(p) = 2*4^(1/p) - 4 for the first pair:", all(bool((SR(D((1, 0), (0, 1), p)) - (2 * 4 ^ (1 / SR(p)) - 4)).simplify_full() == 0) for p in [1, 3 / 2, 3, 4]))
x3 = (3, -4)
print("[PASS] ||(3,-4)||_1, _2, _inf = 7, 5, 4:", [pn(x3, 1), pn(x3, 2), pn(x3, oo)] == [7, 5, 4])
print("wrote", os.path.relpath(OUT))
