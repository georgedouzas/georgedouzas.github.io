# Figure: the Hopf fibration SU(2) = S^3 -> S^2. The fibre over a point of S^2 is a great circle of S^3; after
# stereographic projection of S^3 to R^3 the fibres over a parallel of S^2 form a torus of linked circles
# (Villarceau circles), fibres over different parallels form nested tori. Output: figures/hopf.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hopf.png")
t = var("t", domain="real")
def fibre(theta, phi):
    """The Hopf circle over the point (theta, phi) of S^2, as a curve in S^3 (two complex numbers a, b),
    then stereographically projected from (0, 0, 0, 1) to R^3."""
    a = cos(theta / 2) * exp(I * (phi / 2 + t)); b = sin(theta / 2) * exp(I * (-phi / 2 + t))
    x1, x2, x3, x4 = a.real(), a.imag(), b.real(), b.imag()
    return vector([x1 / (1 - x4), x2 / (1 - x4), x3 / (1 - x4)])
scene = None
colors = ["royalblue", "darkgreen", "firebrick", "darkorange"]
for k, theta in enumerate([pi / 6, pi / 3, pi / 2, 2 * pi / 3]):
    for j in range(8):
        c = parametric_plot3d(fibre(theta, j * pi / 4), (t, 0, 2 * pi), color=colors[k], thickness=2)
        scene = c if scene is None else scene + c
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.1, aspect_ratio=1)
# checks the text states: the Hopf map (a, b) -> (2 Re(a conj b), 2 Im(a conj b), |a|^2 - |b|^2) sends the fibre
# to the single point (sin th cos ph, sin th sin ph, cos th), and is independent of the fibre parameter t
th, ph = var("th ph", domain="real")
# a = c e^{i alpha}, b = s e^{i beta} in rectangular form, with alpha = phi/2 + t, beta = -phi/2 + t
c, s, al, be = cos(th / 2), sin(th / 2), ph / 2 + t, -ph / 2 + t
a_re, a_im, b_re, b_im = c * cos(al), c * sin(al), s * cos(be), s * sin(be)
# 2 a conj(b) = 2 (a_re + i a_im)(b_re - i b_im)
hopf = vector([2 * (a_re * b_re + a_im * b_im), 2 * (a_im * b_re - a_re * b_im), a_re**2 + a_im**2 - b_re**2 - b_im**2])
target = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
print("[PASS] Hopf map sends the fibre over (theta, phi) to that point of S^2, for every t:",
      bool(all(e.trig_reduce().simplify_full() == 0 for e in (hopf - target))))
print("[PASS] |a|^2 + |b|^2 = 1 on the fibre:", bool((a_re**2 + a_im**2 + b_re**2 + b_im**2).simplify_full() == 1))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the same fibres in threejs, so the reader can rotate the picture and see the
# linking. Added: the fibre over the north pole (the unit circle in the xy-plane, black) and over the south pole (the
# z-axis, grey, closed through the point at infinity), around which all other fibres wind. The checks: any two of the
# drawn fibres are linked once (Gauss linking integral, numerically), and the projected fibre over (theta, phi) is a
# round circle (its points are coplanar and equidistant from the circumcentre of three of them).
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
_iscene = None
for k, theta in enumerate([pi / 6, pi / 3, pi / 2, 2 * pi / 3]):
    for j in range(8):
        c = parametric_plot3d(fibre(theta, j * pi / 4), (t, 0, 2 * pi), color=colors[k], thickness=3, plot_points=120)
        _iscene = c if _iscene is None else _iscene + c
_iscene += parametric_plot3d((cos(t), sin(t), 0), (t, 0, 2 * pi), color="black", thickness=3)
_iscene += line3d([(0, 0, -4), (0, 0, 4)], color="grey", thickness=3)
threejs_html("hopf", _iscene, frame=False)
def _pts(theta, phi, n=400):
    fs = [fast_callable(e, vars=[t], domain=RDF) for e in fibre(theta, phi)]
    return [vector(RDF, [f(2 * float(pi) * k / n) for f in fs]) for k in range(n)]
def _linking(P, Q):
    """Gauss linking integral (1/4 pi) sum (dP x dQ).(P - Q)/|P - Q|^3 for two closed polygons."""
    s = 0.0
    for i in range(len(P)):
        dP = P[(i + 1) % len(P)] - P[i]; Pm = (P[(i + 1) % len(P)] + P[i]) / 2
        for k in range(len(Q)):
            dQ = Q[(k + 1) % len(Q)] - Q[k]; Qm = (Q[(k + 1) % len(Q)] + Q[k]) / 2
            r = Pm - Qm
            s += dP.cross_product(dQ).dot_product(r) / r.norm() ** 3
    return s / (4 * pi.n())
_F1, _F2, _F3 = _pts(pi / 3, 0, 200), _pts(pi / 3, pi / 2, 200), _pts(2 * pi / 3, pi / 4, 200)
_l12, _l13 = _linking(_F1, _F2), _linking(_F1, _F3)
print("linking numbers of two fibres over theta = pi/3, and over theta = pi/3 and 2 pi/3:", round(_l12, 4), round(_l13, 4))
print("[PASS] distinct Hopf fibres are linked once (|lk| = 1):", bool(abs(abs(_l12) - 1) < 1e-2 and abs(abs(_l13) - 1) < 1e-2))
_P = _pts(pi / 3, pi / 4, 60); _p0, _p1, _p2 = _P[0], _P[20], _P[40]
_u, _w = _p1 - _p0, _p2 - _p0; _n = _u.cross_product(_w)
_cen = _p0 + (_u.dot_product(_u) * _w.cross_product(_n) + _w.dot_product(_w) * _n.cross_product(_u)) / (2 * _n.dot_product(_n))
_rad = [(p - _cen).norm() for p in _P]
print("[PASS] the projected fibre is a round circle (coplanar, equidistant from the circumcentre):",
      bool(max(_rad) - min(_rad) < 1e-9 and max(abs((p - _p0).dot_product(_n)) for p in _P) < 1e-9))
print("wrote figures/interactive/hopf.html")
