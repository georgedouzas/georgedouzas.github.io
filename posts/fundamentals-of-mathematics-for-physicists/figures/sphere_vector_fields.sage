# Figure: two vector fields on the sphere, defined in the spherical chart with SageManifolds and plotted on the
# embedded sphere: the rotation field d/dphi (blue, tangent to the parallels, vanishing at the poles) and the
# gradient of the height, -sin(theta) d/dtheta (red, along the meridians towards the north pole, vanishing at both
# poles). Output: figures/sphere_vector_fields.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sphere_vector_fields.png")
E3 = Manifold(3, "R^3"); C = E3.chart("X Y Z")
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
emb = A.diff_map(E3, {(sph, C): [sin(th)*cos(ph), sin(th)*sin(ph), cos(th)]}, name="E")
rot = A.vector_field(0, 1, name="rot")                 # d/dphi
grad = A.vector_field(-sin(th), 0, name="gradZ")       # -sin(theta) d/dtheta
scene = sphere((0, 0, 0), 0.99, color="whitesmoke", opacity=0.45)
scene += rot.plot(chart=C, mapping=emb, ranges={th: (0.3, 2.9), ph: (0.05, 2*pi - 0.05)}, number_values=7, scale=0.35, color="royalblue", label_axes=False)
scene += grad.plot(chart=C, mapping=emb, ranges={th: (0.3, 2.9), ph: (0.05, 2*pi - 0.05)}, number_values=7, scale=0.35, color="firebrick", label_axes=False)
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.3, aspect_ratio=1)
br = rot.bracket(grad)
print("[PASS] the two fields commute, [d/dphi, -sin(theta) d/dtheta] = 0:", br == 0)
E = vector([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)])
print("[PASS] the ambient length of d/dphi is sin(theta), vanishing at the poles:", bool((E.diff(ph).norm()**2).simplify_full() == sin(th)**2))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the same two fields on the embedded sphere in Sage's threejs viewer, so that the
# reader can rotate the sphere and look at the poles, where both fields shrink to zero, and at the equator, where the
# rotation field has unit length and the gradient field is longest; the two fields commute (checked above)
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
_scene = sphere((0, 0, 0), 0.99, color="whitesmoke", opacity=0.45)
_scene += rot.plot(chart=C, mapping=emb, ranges={th: (0.3, 2.9), ph: (0.05, 2*pi - 0.05)}, number_values=9, scale=0.35, color="royalblue", label_axes=False)
_scene += grad.plot(chart=C, mapping=emb, ranges={th: (0.3, 2.9), ph: (0.05, 2*pi - 0.05)}, number_values=9, scale=0.35, color="firebrick", label_axes=False)
_scene += parametric_plot3d([cos(ph), sin(ph), 0], (ph, 0, 2*pi), color="black", thickness=1.5)   # the equator
_scene += point3d((0, 0, 1), size=14, color="black") + point3d((0, 0, -1), size=14, color="black")   # the two poles
threejs_html("sphere_vector_fields", _scene, frame=False)
print("wrote figures/interactive/sphere_vector_fields.html")
