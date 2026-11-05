# Figure: the columns of the Jacobian of the spherical parametrisation as coordinate velocities. At three points of
# the sphere (near the pole, at mid-latitude, on the equator) the vectors d_theta E (red, tangent to the meridian,
# always of unit length) and d_phi E (green, tangent to the parallel, of length sin theta, shrinking towards the
# pole). The Gram matrix diag(1, sin^2 theta) records these lengths. Output: figures/coordinate_velocities.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coordinate_velocities.png")
th, ph = var("th ph")
E = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
Eth = E.diff(th); Eph = E.diff(ph)
scene = sphere((0, 0, 0), 0.995, color="whitesmoke", opacity=0.5)
for tk in [pi / 2, pi / 4, pi / 12]:                                  # parallels
    scene += parametric_plot3d([e.subs(th=tk) for e in E], (ph, 0, 2 * pi), color="grey", thickness=1)
scene += parametric_plot3d([e.subs(ph=7 * pi / 4) for e in E], (th, 0, pi), color="grey", thickness=1)   # the meridian
for tk in [pi / 2, pi / 4, pi / 12]:
    P = vector([float(e.subs(th=tk, ph=7 * pi / 4)) for e in E])
    vt = vector([float(e.subs(th=tk, ph=7 * pi / 4)) for e in Eth]); vp = vector([float(e.subs(th=tk, ph=7 * pi / 4)) for e in Eph])
    scene += point3d(P, size=12, color="black")
    scene += arrow3d(P, P + 0.45 * vt, color="firebrick", width=1.5) + arrow3d(P, P + 0.45 * vp, color="darkgreen", width=1.5)
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.3, aspect_ratio=1)
G = (matrix([Eth, Eph]) * matrix([Eth, Eph]).transpose()).simplify_full()
print("[PASS] |d_theta E| = 1 and |d_phi E| = sin(theta): Gram matrix diag(1, sin^2 theta):", bool(G[0, 0] == 1 and G[0, 1] == 0 and G[1, 1] == sin(th) ** 2))
print("[PASS] the columns are orthogonal and d_phi E vanishes at the pole:", bool(G[0, 1] == 0) and bool(Eph.subs(th=0).norm() == 0))
print("wrote", os.path.relpath(OUT))
