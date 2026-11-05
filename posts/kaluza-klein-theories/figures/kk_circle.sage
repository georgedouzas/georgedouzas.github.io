# Figure: the Kaluza-Klein ansatz in two dimensions, ds^2 = dx^2 + phi(x)^2 (dy + A(x) dx)^2 with y ~ y + 2 pi (the
# Riemannian toy: the circle block is taken positive so that the metric embeds as a surface; the five-dimensional
# ansatz of the text has it negative, ds^2 = g - phi^2 (dy + A dx)^2, which flips the sign of R), drawn
# as the surface of revolution of radius phi(x) in orthographic projection. The gray ellipses are the circle fibres
# of constant x (the scalar phi is their radius); the red curves are the horizontal curves dy + A dx = 0, closed
# where A = 0 (left half) and winding around the surface where A != 0 (right half); dashed parts are on the far
# side. With SageManifolds the curvature of this metric is R = -2 phi''/phi, independent of A: in two dimensions A
# is pure gauge (y -> y - int A dx removes it) and only phi curves the surface. Output: figures/kk_circle.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kk_circle.svg")
M = Manifold(2, "M"); X = M.chart("x y"); x, y = X[:]
phi = function("phi")(x); A = function("A")(x)
g = M.metric("g"); g[0, 0] = 1 + phi^2 * A^2; g[0, 1] = phi^2 * A; g[1, 1] = phi^2
Rs = g.ricci_scalar().expr().simplify_full()
print("R of dx^2 + phi^2 (dy + A dx)^2 =", Rs)
print("[PASS] R = -2 phi''/phi independent of A:", bool((Rs + 2 * diff(phi, x, 2) / phi).simplify_full() == 0))
phi_n = lambda s: 1 + 0.35 * sin(s)
A_n = lambda s: 0.6 * (1 + tanh(4 * s)) / 2
tilt = 0.22                                   # the ellipses' minor semi-axis, a hint of perspective
s, v = var("s v")
p = Graphics()
p += plot(phi_n(s), (s, -3, 3), color="steelblue", thickness=1.5) + plot(-phi_n(s), (s, -3, 3), color="steelblue", thickness=1.5)
for x0 in (-2.5, -1.25, 0, 1.25, 2.5):
    p += parametric_plot((x0 + tilt * sin(v), phi_n(x0) * cos(v)), (v, 0, pi), color="gray", thickness=1)
    p += parametric_plot((x0 + tilt * sin(v), phi_n(x0) * cos(v)), (v, pi, 2 * pi), color="gray", thickness=1, linestyle=":")
def yhor(q, y0): return y0 - numerical_integral(lambda u: A_n(u), -3, q)[0]
for y0 in [float(k * pi / 3) for k in range(6)]:
    pts = [(q, yhor(q, y0)) for q in srange(-3, 3.001, 0.05)]
    seg = []; front = None
    for (q, yy) in pts:
        f = sin(yy) >= 0
        if front is None or f == front: seg.append((q + tilt * sin(yy), phi_n(q) * cos(yy)))
        else:
            p += line(seg, color="firebrick", thickness=1.8 if front else 1, linestyle="-" if front else ":"); seg = [(q + tilt * sin(yy), phi_n(q) * cos(yy))]
        front = f
    p += line(seg, color="firebrick", thickness=1.8 if front else 1, linestyle="-" if front else ":")
p += text("$A = 0$: horizontal curves close", (-1.6, 1.75), fontsize=9, color="firebrick")
p += text("$A \\neq 0$: they wind", (1.6, 1.75), fontsize=9, color="firebrick")
p += text("$\\phi(x)$ = radius of the fibre", (0, -1.75), fontsize=9, color="steelblue")
p += arrow((-3.2, 0), (3.4, 0), color="black", width=0.6, arrowsize=2); p += text("$x$", (3.55, 0), fontsize=10)
p.axes(False); p.set_aspect_ratio(1); p.ymin(-2); p.ymax(2.1)
p.save(OUT, figsize=[9, 4])
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the same surface in three dimensions, (x, phi(x) cos y, phi(x) sin y), with the
# circle fibres of constant x (gray) and the horizontal curves dy + A dx = 0 (red), which wind once A is switched on
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
_u, _w = var("_u _w")
_iscene = parametric_plot3d((_u, phi_n(_u) * cos(_w), phi_n(_u) * sin(_w)), (_u, -3, 3), (_w, 0, 2 * pi), color="lightsteelblue", opacity=0.55, plot_points=[60, 40])
for x0 in (-2.5, -1.25, 0, 1.25, 2.5):
    _iscene += parametric_plot3d((x0, phi_n(x0) * cos(_w), phi_n(x0) * sin(_w)), (_w, 0, 2 * pi), color="dimgray", thickness=2)
_qs = [float(q) for q in srange(-3, 3.001, 0.05)]
_Aint = [numerical_integral(lambda u: A_n(u), -3, q)[0] for q in _qs]
for y0 in [float(k * pi / 3) for k in range(6)]:
    _iscene += line3d([(q, float(phi_n(q)) * cos(y0 - a), float(phi_n(q)) * sin(y0 - a)) for q, a in zip(_qs, _Aint)], color="firebrick", thickness=3)
threejs_html("kk_circle", _iscene, frame=False, aspect_ratio=1)
print("wrote figures/interactive/kk_circle.html")
