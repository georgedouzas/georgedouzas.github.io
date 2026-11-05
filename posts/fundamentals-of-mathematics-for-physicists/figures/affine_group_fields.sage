# Figure: left-invariant vector fields on a two-dimensional non-abelian Lie group, the affine group of the line
# {x -> a x + b, a > 0}, with coordinates (a, b) on the half-plane. Left multiplication by (a0, b0) sends (a, b) to
# (a0 a, a0 b + b0); the left-invariant fields through the identity (1, 0) are X = a d/da (blue) and Y = a d/db
# (red), and their bracket [X, Y] = Y is computed with SageManifolds. Output: figures/affine_group_fields.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "affine_group_fields.svg")
G = Manifold(2, "Aff")
C = G.chart("a:(0,+oo) b"); a, b = C[:]
X = G.vector_field(a, 0, name="X"); Y = G.vector_field(0, a, name="Y")
br = X.bracket(Y)
aa, bb = var("aa bb", domain="real")
p = plot_vector_field((aa, 0), (aa, 0.3, 3), (bb, -1.5, 1.5), color="royalblue", plot_points=8)
p += plot_vector_field((0, aa), (aa, 0.3, 3), (bb, -1.5, 1.5), color="firebrick", plot_points=8)
p += point((1, 0), color="black", size=50) + text("identity $(1,0)$", (1.0, -0.25), fontsize=9)
p += text(r"$X=a\,\partial_a$ (blue), $Y=a\,\partial_b$ (red): $[X,Y]=Y$", (1.65, 1.85), fontsize=10)
p.axes_labels([r"$a$", r"$b$"])
p.save(OUT, figsize=[5.5, 4.2])
# left invariance: the pushforward of X by L_(a0,b0): (a, b) -> (a0 a, a0 b + b0) equals X at the image point
a0, b0 = var("a0 b0", domain="real")
Lmap = G.diff_map(G, {(C, C): [a0 * a, a0 * b + b0]}, name="L")
J = jacobian([a0 * a, a0 * b + b0], (a, b))
pushX = J * vector([a, 0]); pushY = J * vector([0, a])
print("[PASS] X and Y are left-invariant: dL(X) = X at the image point, dL(Y) = Y at the image point:", bool(pushX == vector([a0 * a, 0])) and bool(pushY == vector([0, a0 * a])))
print("[PASS] [X, Y] = Y (a non-abelian two-dimensional Lie algebra):", br == Y)
print("wrote", os.path.relpath(OUT))
