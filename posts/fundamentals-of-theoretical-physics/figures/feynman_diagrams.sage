# Figure: the lowest Feynman diagrams of the quartic theory L = (1/2)(d phi)^2 - (1/2) m^2 phi^2 - (lambda/4!) phi^4,
# drawn with Sage graphics, each with its symmetry factor: the tree-level vertex (four external lines, factor 1), the
# one-loop self-energy (tadpole, 1/2), the one-loop s-channel bubble (1/2) and the figure-eight vacuum diagram (1/8).
# The symmetry factors are computed by brute-force enumeration of Wick contractions; the one-loop integrals by Gamma
# functions. Output: figures/feynman_diagrams.svg.
import os, itertools
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feynman_diagrams.svg")
lw = 1.6; vc = "black"
def ext(a, b, lab=None, where=None):
    g = line([a, b], color="royalblue", thickness=lw)
    if lab:
        g += text(lab, where if where else b, fontsize=9, color="black")
    return g
def vtx(p):
    return point(p, size=40, color=vc)
tree = ext((-1, 1), (0, 0), r"$p_1$", (-1.15, 1.1)) + ext((-1, -1), (0, 0), r"$p_2$", (-1.15, -1.1)) + ext((0, 0), (1, 1), r"$p_3$", (1.15, 1.1)) + ext((0, 0), (1, -1), r"$p_4$", (1.15, -1.1)) + vtx((0, 0))
tree += text(r"tree: $-i\lambda$, factor $1$", (0, -1.6), fontsize=9, color="black")
tad = ext((-1.5, 0), (0, 0), r"$p$", (-1.4, -0.2)) + ext((0, 0), (1.5, 0), r"$p$", (1.4, -0.2)) + circle((0, 0.55), 0.55, color="firebrick", thickness=lw) + vtx((0, 0))
tad += text(r"$\ell$", (0, 1.25), fontsize=9, color="firebrick") + text(r"self-energy, factor $\frac{1}{2}$", (0, -1.6), fontsize=9, color="black")
bub = ext((-1.9, 1), (-0.7, 0), r"$p_1$", (-2.0, 1.15)) + ext((-1.9, -1), (-0.7, 0), r"$p_2$", (-2.0, -1.15)) + ext((0.7, 0), (1.9, 1), r"$p_3$", (2.0, 1.15)) + ext((0.7, 0), (1.9, -1), r"$p_4$", (2.0, -1.15))
bub += circle((0, 0), 0.7, color="firebrick", thickness=lw) + vtx((-0.7, 0)) + vtx((0.7, 0))
bub += text(r"$\ell$", (0, 0.95), fontsize=9, color="firebrick") + text(r"$\ell+p$", (0, -0.95), fontsize=9, color="firebrick")
bub += text(r"$s$-channel bubble, factor $\frac{1}{2}$", (0, -1.6), fontsize=9, color="black")
eight = circle((-0.55, 0), 0.55, color="firebrick", thickness=lw) + circle((0.55, 0), 0.55, color="firebrick", thickness=lw) + vtx((0, 0))
eight += text(r"vacuum figure eight, factor $\frac{1}{8}$", (0, -1.6), fontsize=9, color="black")
for G in (tree, tad, bub, eight):
    G.axes(False); G.set_aspect_ratio(1)
g = graphics_array([tree, tad, bub, eight], 1, 4)
fig = g.matplotlib(figsize=[11, 3]); fig.savefig(OUT, bbox_inches="tight")

# 1. symmetry factors: count the Wick contractions of fields ('x', i) (external points) and ('v', a, leg) (legs of vertex a)
def matchings(pts):
    if not pts:
        yield []
        return
    a = pts[0]
    for b in pts[1:]:
        rest = [p for p in pts if p != a and p != b]
        for m in matchings(rest):
            yield [(a, b)] + m
def node(f):
    return f[:2]
def multigraph(m):
    return sorted(tuple(sorted((node(a), node(b)))) for a, b in m)
def count(nexts, nverts, accept):
    fields = [("x", i) for i in range(nexts)] + [("v", a, l) for a in range(nverts) for l in range(4)]
    return sum(1 for m in matchings(fields) if accept(multigraph(m)))
fac = lambda N, nv: QQ(N) / (factorial(nv) * factorial(4) ** nv)
s_eight = fac(count(0, 1, lambda e: e == [(("v", 0), ("v", 0))] * 2), 1)
s_tree = fac(count(4, 1, lambda e: all(("v", 0) in p and p[0] != p[1] for p in e)), 1)
s_tad = fac(count(2, 1, lambda e: sorted(e) == sorted([(("v", 0), ("x", 0)), (("v", 0), ("x", 1)), (("v", 0), ("v", 0))])), 1)
def is_s_bubble(e):
    ok = {}
    for p in e:
        if p[0][0] == "v" and p[1][0] == "x":
            ok.setdefault(p[0][1], []).append(p[1][1])
    across = e.count((("v", 0), ("v", 1)))
    return across == 2 and sorted(map(sorted, ok.values())) in ([[0, 1], [2, 3]],)
s_bub = fac(count(4, 2, is_s_bubble), 2)
print("[PASS] symmetry factors from counting Wick contractions: figure eight 1/8, tree vertex 1, one-loop self-energy 1/2, s-channel bubble 1/2:",
      (s_eight, s_tree, s_tad, s_bub) == (QQ(1) / 8, 1, QQ(1) / 2, QQ(1) / 2))
s_sunset = fac(count(2, 2, lambda e: sorted(e) == sorted([(("v", 0), ("x", 0)), (("v", 1), ("x", 1))] + [(("v", 0), ("v", 1))] * 3)), 2) * 2
print("[PASS] the two-loop sunset self-energy has symmetry factor 1/6 (both assignments of the external points counted):", s_sunset == QQ(1) / 6)
# 2. the one-loop integral by Gamma functions
l, D = var("l D"); assume(D > 0)
def radial(d, n):
    Om = 2 * pi ** (QQ(d) / 2) / gamma(QQ(d) / 2)
    return Om * integrate(l ** (d - 1) / (l ** 2 + D) ** n, l, 0, oo) / (2 * pi) ** d
def formula(d, n):
    return gamma(n - QQ(d) / 2) / ((4 * pi) ** (QQ(d) / 2) * gamma(n)) * D ** (QQ(d) / 2 - n)
cases = [(1, 1), (1, 2), (2, 2), (3, 2), (3, 3), (4, 3), (5, 4)]
print("[PASS] int d^d l/(2 pi)^d 1/(l^2 + Delta)^n = Gamma(n - d/2)/((4 pi)^{d/2} Gamma(n)) Delta^{d/2 - n} by radial integration for (d, n) in", cases, ":",
      all(bool((radial(d, n) / formula(d, n)).simplify_full() == 1) for d, n in cases))
ep = var("ep")
ser = gamma(2 - (4 - ep) / 2).series(ep, 1).truncate()
print("[PASS] at d = 4 - eps the bubble coefficient Gamma(2 - d/2) = 2/eps - gamma_E + O(eps):", bool(ser == 2 / ep - euler_gamma))
A, B, xf = var("A B xf")
u = xf * A + (1 - xf) * B; F = 1 / ((B - A) * u)          # an antiderivative of 1/u^2 (A != B)
print("[PASS] Feynman parameters: 1/(A B) = int_0^1 dx 1/(x A + (1 - x) B)^2:",
      bool((diff(F, xf) - 1 / u ** 2).simplify_full() == 0) and bool((F.subs(xf=1) - F.subs(xf=0) - 1 / (A * B)).simplify_full() == 0)
      and bool(integrate(1 / (xf * 2 + (1 - xf) * 5) ** 2, xf, 0, 1) == QQ(1) / 10))
print("wrote", os.path.relpath(OUT))
