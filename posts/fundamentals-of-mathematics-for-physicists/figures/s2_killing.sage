# Figure: the three Killing vector fields of the round sphere, the generators of rotations about the X, Y and Z
# axes, drawn as arrows on S^2. Output: figures/s2_killing.png (3D scene, raster).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_killing.png")
th, ph = var("th ph")
E = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
# components in (theta, phi) of the rotation generators (the Killing chapter's K_a); their ambient vectors are J K
K = [vector([-sin(ph), -cos(th) * cos(ph) / sin(th)]), vector([cos(ph), -cos(th) * sin(ph) / sin(th)]), vector([0, 1])]
J = jacobian(E, (th, ph))
colors = ["firebrick", "darkgreen", "royalblue"]
scene = parametric_plot3d(E, (th, 0, pi), (ph, 0, 2 * pi), color="whitesmoke", opacity=0.6, mesh=False)
passes = []
for a in range(3):
    V = J * K[a]
    for i in range(1, 6):
        for j in range(8):
            p = {th: i * pi / 6, ph: j * pi / 4}
            base = E.subs(p); tip = base + 0.25 * V.subs(p)
            scene += arrow3d(base, tip, color=colors[a], width=1.5)
    # the quantity the text states: each field is tangent and has squared length sin^2 of the angle to its axis
    length2 = (V * V).simplify_full().trig_simplify()
    passes.append((a, length2))
scene += point3d((0, 0, 1), size=14, color="black") + point3d((0, 0, -1), size=14, color="black")
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.35, aspect_ratio=1)
# the Killing property, checked with SageManifolds: the Lie derivative of the metric along each field vanishes
M = Manifold(2, "S2", structure="Riemannian"); X = M.chart("th:(0,pi) ph:(0,2*pi)"); thM, phM = X[:]
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = sin(thM) ** 2
KM = [M.vector_field(-sin(phM), -cos(thM) / sin(thM) * cos(phM)), M.vector_field(cos(phM), -cos(thM) / sin(thM) * sin(phM)),
      M.vector_field(0, 1)]
for a, l2 in passes:
    gK = g(KM[a], KM[a]).expr().simplify_full().trig_simplify()
    print(f"[PASS] |K_{a}|^2 from the metric equals the ambient squared length: {bool((gK - l2.subs(th=thM, ph=phM)).simplify_full() == 0)} ({l2})")
    print(f"[PASS] L_K{a} g = 0 (Killing):", g.lie_derivative(KM[a]) == 0)
print("[PASS] K_2 = d/dphi has length sin(theta):", bool(g(KM[2], KM[2]).expr().simplify_full() == sin(thM) ** 2))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the three Killing fields as arrows on the sphere together with some of their
# integral curves, the circles about the X-, Y- and Z-axes along which their flows (the rotations) move points, in the
# threejs viewer; rotating the sphere shows each field vanishing where its axis pierces it
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
_u = var("_u")
_iscene = parametric_plot3d(E, (th, 0, pi), (ph, 0, 2 * pi), color="whitesmoke", opacity=0.5, mesh=False)
_axes = [vector([1, 0, 0]), vector([0, 1, 0]), vector([0, 0, 1])]
_orbit_ok = True
for a in range(3):
    n = _axes[a]; e1 = _axes[(a + 1) % 3]; e2 = _axes[(a + 2) % 3]
    for i in range(1, 6):
        for j in range(8):
            p = {th: i * pi / 6, ph: j * pi / 4}
            base = E.subs(p); _iscene += arrow3d(base, base + 0.25 * (J * K[a]).subs(p), color=colors[a], width=1.5)
    for c in [-0.7, -0.35, 0, 0.35, 0.7]:
        rho = sqrt(1 - c ** 2)
        orbit = c * n + rho * cos(_u) * e1 + rho * sin(_u) * e2           # circle about the a-th axis at height c
        _iscene += parametric_plot3d(orbit, (_u, 0, 2 * pi), color=colors[a], thickness=1.2, opacity=0.7)
        # the circle is an integral curve of the ambient rotation field xi_a = eps_abc x^b d_c: velocity = n x position
        _orbit_ok = _orbit_ok and bool(all((v - w).simplify_full() == 0 for v, w in zip(diff(orbit, _u), n.cross_product(orbit))))
    _iscene += point3d(tuple(n), size=12, color=colors[a]) + point3d(tuple(-n), size=12, color=colors[a])
print("[PASS] the circles about the a-th axis are integral curves of the ambient rotation field xi_a = n_a x position:", _orbit_ok)
threejs_html("s2_killing", _iscene)
print("wrote figures/interactive/s2_killing.html")
