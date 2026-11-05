# Figure: the cosets of the subgroup H = {0, 2, 4} in Z_6, drawn on the circle of sixth roots of unity (Z_6 as the
# rotations by multiples of 60 degrees). The two cosets H and 1 + H are the two triangles; the quotient Z_6/H = Z_2
# swaps them. Output: figures/coset_z6.svg (2D, vector).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coset_z6.svg")
pts = [(cos(k*pi/3), sin(k*pi/3)) for k in range(6)]
H = [0, 2, 4]; H1 = [1, 3, 5]
p = circle((0, 0), 1, color="gray", linestyle="--")
p += polygon([pts[k] for k in H], color="royalblue", alpha=0.25) + polygon([pts[k] for k in H1], color="firebrick", alpha=0.25)
for k in range(6):
    p += point(pts[k], size=60, color="royalblue" if k in H else "firebrick")
    p += text(str(k), (1.18*pts[k][0], 1.18*pts[k][1]), fontsize=12, color="black")
p += text("H = {0, 2, 4}", (0, -1.45), color="royalblue", fontsize=11) + text("1 + H = {1, 3, 5}", (0, -1.65), color="firebrick", fontsize=11)
p.set_aspect_ratio(1); p.axes(False)
p.save(OUT, figsize=5)
G = Zmod(6); Hs = {G(0), G(2), G(4)}
print("[PASS] 1 + H = {1, 3, 5}:", {g + G(1) for g in Hs} == {G(1), G(3), G(5)})
print("[PASS] the two cosets partition Z_6:", Hs | {g + G(1) for g in Hs} == set(G))
print("[PASS] quotient table: (1+H) + (1+H) = H:", {a + b for a in {g + G(1) for g in Hs} for b in {g + G(1) for g in Hs}} == Hs)
print("wrote", os.path.relpath(OUT))
