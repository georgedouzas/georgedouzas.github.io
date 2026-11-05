# Figure: tangent vectors on the sphere. At a point p the tangent plane T_p S^2, the coordinate basis vectors
# d/dtheta and d/dphi (the velocities of the coordinate curves), and a general tangent vector X = X^th d/dth + X^ph d/dph.
# Output: figures/tangent_vectors.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tangent_vectors.png")
th, ph = var("th ph")
E = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
J = jacobian(E, (th, ph))
p = {th: pi / 3, ph: pi / 4}
P = E.subs(p); e_th = J.column(0).subs(p); e_ph = J.column(1).subs(p)
scene = parametric_plot3d(E, (th, 0, pi), (ph, 0, 2 * pi), color="whitesmoke", opacity=0.55, mesh=False)
u, v = var("u v")
scene += parametric_plot3d(list(P + u * e_th + v * e_ph), (u, -0.6, 0.6), (v, -0.7, 0.7), color="lightsteelblue", opacity=0.45)
scene += arrow3d(P, P + 0.6 * e_th, color="firebrick", width=2) + arrow3d(P, P + 0.6 * e_ph, color="darkgreen", width=2)
X = 0.4 * e_th + 0.5 * e_ph
scene += arrow3d(P, P + X, color="royalblue", width=2.5)
scene += parametric_plot3d(list(E.subs(ph=pi / 4)), (th, 0, pi), color="firebrick", thickness=1.5)          # the theta coordinate curve
scene += parametric_plot3d(list(E.subs(th=pi / 3)), (ph, 0, 2 * pi), color="darkgreen", thickness=1.5)      # the phi coordinate curve
scene += point3d(tuple(P), size=16, color="black")
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.3, aspect_ratio=1)
print("[PASS] the coordinate basis vectors are tangent (orthogonal to the position vector):", bool(e_th * P == 0 and e_ph * P == 0))
print("[PASS] |d/dtheta| = 1 and |d/dphi| = sin(theta) at the point:", bool((e_th * e_th).simplify_full() == 1 and (e_ph * e_ph).simplify_full() == sin(pi / 3) ** 2))
print("wrote", os.path.relpath(OUT))
