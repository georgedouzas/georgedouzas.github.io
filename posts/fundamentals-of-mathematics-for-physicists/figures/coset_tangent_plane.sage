# Figure: the coset decomposition so(3) = r + k seen at the north pole of S^2 = SO(3)/SO(2): the isotropy
# generator L_3 rotates the tangent plane (the plane k) about the pole, while L_1 and L_2 move the pole along the
# two tangent directions. Output: figures/coset_tangent_plane.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coset_tangent_plane.png")
th, ph, t = var("th ph t")
E = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
scene = parametric_plot3d(E, (th, 0, pi), (ph, 0, 2 * pi), color="whitesmoke", opacity=0.5, mesh=False)
# the tangent plane k at the north pole
u, v = var("u v")
scene += parametric_plot3d((u, v, 1), (u, -0.9, 0.9), (v, -0.9, 0.9), color="lightsteelblue", opacity=0.45)
pole = vector(SR, [0, 0, 1])
L1, L2, L3 = (matrix(SR, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(SR, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(SR, [[0,-1,0],[1,0,0],[0,0,0]]))
# L_1, L_2 move the pole: the velocities L_a . pole are tangent vectors spanning k
for La, col, lab in [(L1, "firebrick", "L1"), (L2, "darkgreen", "L2")]:
    vel = La * pole
    scene += arrow3d(pole, pole + 0.7 * vel, color=col, width=2)
# L_3 fixes the pole and rotates the plane: draw the orbit of a tangent vector under exp(t L3) as a circle in k
w0 = vector([0.6, 0, 1])
orbit = parametric_plot3d((0.6 * cos(t), 0.6 * sin(t), 1), (t, 0, 2 * pi), color="royalblue", thickness=3)
scene += orbit + arrow3d((0.6, 0, 1), (0.6, 0.35, 1), color="royalblue", width=2)
# the orbits of the pole under exp(t L1), exp(t L2): great circles through the pole (the geodesics in k directions)
for La, col in [(L1, "firebrick"), (L2, "darkgreen")]:
    scene += parametric_plot3d(list((t * La).exp() * pole), (t, -1.2, 1.2), color=col, thickness=2)
scene += point3d(tuple(pole), size=16, color="black")
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.3, aspect_ratio=1)
# the quantities the text states
print("[PASS] L3 fixes the pole (isotropy):", bool(L3 * pole == 0))
print("[PASS] L1 . pole and L2 . pole are tangent (orthogonal to the pole):", bool((L1 * pole) * pole == 0 and (L2 * pole) * pole == 0))
print("[PASS] [L3, L1] = L2 and [L3, L2] = -L1 (L3 rotates k):", bool(L3*L1 - L1*L3 == L2 and L3*L2 - L2*L3 == -L1))
print("wrote", os.path.relpath(OUT))
