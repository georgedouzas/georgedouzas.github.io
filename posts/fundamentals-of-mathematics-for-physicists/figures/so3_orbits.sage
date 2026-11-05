# Figure: orbits of one-parameter subgroups of SO(3) on the sphere. exp(t L_3) moves points along parallels
# (the flow of the Killing field K_3 = d/dphi); exp(t L_1) along great circles about the X-axis (the flow of K_1).
# Every point of S^2 lies on the orbit of the pole under SO(3): the action is transitive. Output: so3_orbits.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "so3_orbits.png")
th, ph, t = var("th ph t")
E = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
scene = parametric_plot3d(E, (th, 0, pi), (ph, 0, 2 * pi), color="whitesmoke", opacity=0.5, mesh=False)
L1 = matrix(SR, [[0,0,0],[0,0,-1],[0,1,0]]); L3 = matrix(SR, [[0,-1,0],[1,0,0],[0,0,0]])
def orbit(L, p, col):
    return parametric_plot3d(list(((t * L).exp() * vector(p)).apply_map(lambda e: e.simplify_full())), (t, 0, 2 * pi), color=col, thickness=2)
for k in range(1, 6):                      # exp(t L3) orbits: parallels
    th0 = k * pi / 6
    scene += orbit(L3, (sin(th0), 0, cos(th0)), "royalblue")
for k in range(6):                         # exp(t L1) orbits through points of the equator: great circles about X
    scene += orbit(L1, (cos(k * pi / 6), sin(k * pi / 6), 0), "firebrick")
scene += point3d((0, 0, 1), size=16, color="black")
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.3, aspect_ratio=1)
R = (t * L3).exp()
print("[PASS] exp(t L3) preserves the height z (orbits are parallels):", bool(all(((R * E)[2] - E[2]).simplify_full() == 0 for _ in [0])))
p = vector([0, 0, 1]); g = (pi/3 * L1).exp()
print("[PASS] SO(3) moves the pole to (0, -sin(pi/3), cos(pi/3)):", bool(g * p == vector([0, -sin(pi/3), cos(pi/3)])))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the orbit of the north pole under SO(3) and the cosets g SO(2). For a few
# rotations g = exp(phi L3) exp(theta L1) the points g . o (black) cover the sphere, and the whole coset
# {g exp(s L3)} sends o to the same point while turning the tangent vector (0.3, 0, 0) at the pole into a circle of
# tangent vectors at g . o (orange): the coset is the circle of frames over one point. The isotropy circle at the
# pole itself (blue) is the orbit of that tangent vector under exp(s L3). Shown in the threejs viewer.
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
_s = var("_s")
L2 = matrix(SR, [[0,0,1],[0,0,0],[-1,0,0]])
_iscene = parametric_plot3d(E, (th, 0, pi), (ph, 0, 2 * pi), color="whitesmoke", opacity=0.5, mesh=False)
_iscene += orbit(L1, (0, 0, 1), "firebrick") + orbit(L2, (0, 0, 1), "darkgreen")
_o = vector(SR, [0, 0, 1]); _w = vector(SR, [3/10, 0, 0])
_H = (_s * L3).exp().apply_map(lambda e: e.simplify_full())
_iscene += parametric_plot3d(list(_o + _H * _w), (_s, 0, 2 * pi), color="royalblue", thickness=3)
_iscene += point3d(tuple(_o), size=14, color="black")
_coset_ok = True
for (_th0, _ph0) in [(pi / 4, 0), (pi / 3, 2 * pi / 3), (pi / 2, pi / 3), (2 * pi / 3, 4 * pi / 3), (3 * pi / 4, pi), (pi / 2, 3 * pi / 2)]:
    _g = ((_ph0 * L3).exp() * (_th0 * L1).exp()).apply_map(lambda e: e.simplify_full())
    _p = _g * _o
    _circ = _p + _g * _H * _w                                  # the tip of g h(s) w attached at g . o
    _iscene += point3d(tuple(_p), size=12, color="black")
    _iscene += parametric_plot3d(list(_circ), (_s, 0, 2 * pi), color="darkorange", thickness=2.5)
    for _k in range(4):
        _iscene += arrow3d(tuple(_p), tuple(_circ.subs(_s=_k * pi / 2)), color="darkorange", width=1.2)
    # every element of the coset sends o to g . o, and the turned vector stays tangent at g . o
    _coset_ok = _coset_ok and bool(((_g * _H * _o) - _p).apply_map(lambda e: e.simplify_full()) == 0)
    _coset_ok = _coset_ok and bool(((_g * _H * _w) * _p).simplify_full() == 0)
print("[PASS] each coset g exp(s L3) sends the pole to g . o and turns the tangent plane at g . o:", _coset_ok)
threejs_html("so3_orbits", _iscene)
print("wrote figures/interactive/so3_orbits.html")
