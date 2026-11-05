# Figure: the coset vielbein of S^2 = SU(2)/U(1) as a frame. The dual vectors of e^1 = dtheta and e^2 = sin(theta) dphi,
# namely E_1 = d/dtheta (red) and E_2 = d/dphi / sin(theta) (green), form an orthonormal frame of the round sphere,
# drawn with SageManifolds on the embedded sphere; the third form e^3 = cos(theta) dphi is the U(1) connection and
# has no direction on the sphere. Output: figures/coset_vielbein_sphere.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coset_vielbein_sphere.png")
E3 = Manifold(3, "R^3"); C = E3.chart("X Y Z")
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
emb = A.diff_map(E3, {(sph, C): [sin(th)*cos(ph), sin(th)*sin(ph), cos(th)]}, name="E")
E1 = A.vector_field(1, 0, name="E1"); E2 = A.vector_field(0, 1 / sin(th), name="E2")
scene = sphere((0, 0, 0), 0.99, color="whitesmoke", opacity=0.5)
rng = {th: (0.4, 2.7), ph: (0.05, 2*pi - 0.05)}
scene += E1.plot(chart=C, mapping=emb, ranges=rng, number_values=6, scale=0.3, color="firebrick", label_axes=False)
scene += E2.plot(chart=C, mapping=emb, ranges=rng, number_values=6, scale=0.3, color="darkgreen", label_axes=False)
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.3, aspect_ratio=1)
g = A.metric("g") if False else None
e1 = A.one_form(1, 0, name="e1"); e2 = A.one_form(0, sin(th), name="e2")
print("[PASS] the frame is dual to the vielbein, e^a(E_b) = delta^a_b:", all(bool((ea(Eb).expr() - (1 if a == b else 0)).simplify_full() == 0) for a, ea in enumerate([e1, e2]) for b, Eb in enumerate([E1, E2])))
E = vector([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)])
v1 = E.diff(th); v2 = E.diff(ph) / sin(th)
print("[PASS] the frame vectors have unit ambient length and are orthogonal:", bool((v1 * v1).simplify_full() == 1) and bool((v2 * v2).simplify_full() == 1) and bool((v1 * v2).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
