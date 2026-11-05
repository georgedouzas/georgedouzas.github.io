# Figure: the torus with its two coordinate circles (a around the tube, b around the axis) and the induced metric's
# scale factors: the tube circle has length 2 pi r everywhere, the axis circle has length 2 pi (R + r cos a), largest
# on the outer equator and smallest on the inner one. Output: figures/torus_embedding.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torus_embedding.png")
a, b = var("a b"); R, r = 2, QQ(4)/5
E = ((R + r * cos(a)) * cos(b), (R + r * cos(a)) * sin(b), r * sin(a))
scene = parametric_plot3d(E, (a, 0, 2 * pi), (b, 0, 2 * pi), color="whitesmoke", opacity=0.6, mesh=False)
for bk in [0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3]:                         # tube circles b = const
    scene += parametric_plot3d([e.subs(b=bk) for e in E], (a, 0, 2 * pi), color="firebrick", thickness=1.5)
for ak in [0, pi / 2, pi, 3 * pi / 2]:                                                 # axis circles a = const
    scene += parametric_plot3d([e.subs(a=ak) for e in E], (b, 0, 2 * pi), color="royalblue", thickness=2)
scene.save(OUT, viewer="tachyon", figsize=[7, 5], zoom=1.2, aspect_ratio=1)
# SageManifolds: the induced metric of the embedding and the lengths of the coordinate circles
M = Manifold(2, "T2", structure="Riemannian"); X = M.chart("a:(0,2*pi) b:(0,2*pi)"); aM, bM = X[:]
N = Manifold(3, "R3", structure="Riemannian"); Y = N.chart("x y z"); h = N.metric("h"); h[0, 0] = 1; h[1, 1] = 1; h[2, 2] = 1
Phi = M.diff_map(N, {(X, Y): [(R + r * cos(aM)) * cos(bM), (R + r * cos(aM)) * sin(bM), r * sin(aM)]})
g = Phi.pullback(h)
print("[PASS] induced metric g = r^2 da^2 + (R + r cos a)^2 db^2:", bool(g[0, 0].expr() == r ** 2 and g[0, 1].expr() == 0 and (g[1, 1].expr() - (R + r * cos(aM)) ** 2).simplify_full() == 0))
L_tube = integrate(sqrt(g[0, 0].expr()), aM, 0, 2 * pi)
L_axis = integrate(R + r * cos(aM), bM, 0, 2 * pi)          # sqrt(g_bb) = R + r cos a since R > r > 0
print("[PASS] tube circles have length 2 pi r; axis circles have length 2 pi (R + r cos a):", bool(L_tube == 2 * pi * r) and bool((L_axis - 2 * pi * (R + r * cos(aM))).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the same scene saved with the threejs viewer, so that the torus and its two
# families of coordinate circles can be rotated in the browser
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
threejs_html("torus_embedding", scene)
print("wrote figures/interactive/torus_embedding.html")
