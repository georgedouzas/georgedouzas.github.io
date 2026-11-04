# Exercise exr-atlas-circle-sagemanifolds: the circle S^1 in SageManifolds with its two angle charts, phi in (0, 2 pi)
# on the circle minus (1, 0) and phi' in (-pi, pi) on the circle minus (-1, 0): the inverses land on the circle, the
# two missed points differ, so the two domains cover, and the atlas has two charts.
S1 = Manifold(1, "S^1")
U = S1.open_subset("U"); V = S1.open_subset("V")            # circle minus (1, 0); circle minus (-1, 0)
S1.declare_union(U, V)
cU = U.chart(r"ph:(0,2*pi):\varphi"); cV = V.chart(r"php:(-pi,pi):\varphi'")
ph = cU[0]; php = cV[0]
E2 = Manifold(2, "R^2"); XY = E2.chart("X Y")
incl = S1.diff_map(E2, {(cU, XY): [cos(ph), sin(ph)], (cV, XY): [cos(php), sin(php)]}, name="i")
print("atlas of the circle:", S1.atlas())
for ch in (cU, cV):
    x, y = incl.expr(ch, XY)
    print("inverse of the chart", ch, "lands on X^2 + Y^2 = 1:", bool((x**2 + y**2).simplify_full() == 1))
print("point missed by the first chart (limit phi -> 0 or 2 pi):", (limit(cos(ph), ph=0), limit(sin(ph), ph=0)))
print("point missed by the second chart (limit phi' -> +-pi):", (limit(cos(php), php=pi), limit(sin(php), php=pi)))
print("the missed points differ, so every point of the circle lies in at least one domain:", (1, 0) != (-1, 0))
p = U.point((3*pi/2,), chart=cU, name="p")
print("the point p = (0, -1) has phi =", p.coord(cU)[0], "and lies at (X, Y) =", incl(p).coord(XY))
print("in the second chart the same point has phi' = -pi/2:", (cos(-pi/2), sin(-pi/2)) == (0, -1))
