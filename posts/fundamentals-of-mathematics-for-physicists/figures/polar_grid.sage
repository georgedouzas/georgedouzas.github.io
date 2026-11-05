# Figure: the plane in polar coordinates. The coordinate grid (circles r = const, rays phi = const) shows why the
# metric ds^2 = dr^2 + r^2 dphi^2 has a non-constant component: the rays spread, so a step dphi is longer far from
# the origin, although the plane is flat. Output: figures/polar_grid.svg (2D, vector).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polar_grid.svg")
p = Graphics()
for rk in [0.5, 1, 1.5, 2, 2.5]:
    p += circle((0, 0), rk, color="firebrick")
for k in range(12):
    p += line([(0, 0), (2.7 * cos(k * pi / 6), 2.7 * sin(k * pi / 6))], color="royalblue")
# the coordinate basis vectors at two points: d/dr (unit) and d/dphi (length r)
for (rk, pk) in [(1, pi / 4), (2.2, 3 * pi / 4)]:
    P = (rk * cos(pk), rk * sin(pk))
    p += arrow(P, (P[0] + 0.5 * cos(pk), P[1] + 0.5 * sin(pk)), color="black", width=1.5)
    p += arrow(P, (P[0] - 0.5 * rk * sin(pk), P[1] + 0.5 * rk * cos(pk)), color="darkgreen", width=1.5)
p += text(r"$\partial_r$", (1.0, 1.35), color="black", fontsize=11) + text(r"$\partial_\varphi$", (0.2, 1.25), color="darkgreen", fontsize=11)
p.set_aspect_ratio(1); p.axes(False); p.save(OUT, figsize=5)
M = Manifold(2, "R2", structure="Riemannian"); X = M.chart("r:(0,+oo) ph:(0,2*pi)"); rM, phM = X[:]
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = rM ** 2
print("[PASS] |d/dphi| = r and the plane is flat (Riemann = 0):", bool(g(M.vector_field(0, 1), M.vector_field(0, 1)).expr() == rM ** 2), g.riemann() == 0)
print("wrote", os.path.relpath(OUT))
