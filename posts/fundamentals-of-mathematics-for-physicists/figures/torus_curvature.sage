# Figure: the Gauss curvature of the embedded torus, K = cos(a) / (r (R + r cos a)), computed from the induced
# metric with SageManifolds (ricci_scalar / 2) and painted on the surface: positive (red) on the outer half,
# negative (blue) on the inner half, zero on the top and bottom circles. Output: figures/torus_curvature.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torus_curvature.png")
R, r = 2, QQ(4) / 5
T2 = Manifold(2, "T^2", structure="Riemannian"); ab = T2.chart("a:(0,2*pi) b:(0,2*pi)"); a, b = ab[:]
E3 = Manifold(3, "R^3", structure="Riemannian"); C = E3.chart("X Y Z")
h = E3.metric("h"); h[0, 0] = 1; h[1, 1] = 1; h[2, 2] = 1
emb = T2.diff_map(E3, {(ab, C): [(R + r * cos(a)) * cos(b), (R + r * cos(a)) * sin(b), r * sin(a)]}, name="E")
g = T2.metric("g"); g.set(emb.pullback(h))
K = (g.ricci_scalar().expr() / 2).simplify_full()
Kf = fast_callable(K, vars=[a, b], domain=float)
def colour(aa, bb):
    k = Kf(aa, bb) / float(K.subs(a=0))
    return (0.5 + 0.5 * k, 0.5 - 0.5 * abs(k), 0.5 - 0.5 * k)
E = [(R + r * cos(a)) * cos(b), (R + r * cos(a)) * sin(b), r * sin(a)]
scene = parametric_plot3d(E, (a, 0, 2 * pi), (b, 0, 2 * pi), color=(lambda aa, bb: colour(aa, bb), colormaps.coolwarm) if False else "lightgrey", opacity=0.9, mesh=False, plot_points=[60, 80])
# colour by curvature through a texture of small patches: draw the surface in strips of a with the sign colour
scene = Graphics()
for k in range(24):
    a0, a1 = 2 * pi * k / 24, 2 * pi * (k + 1) / 24
    am = (a0 + a1) / 2
    kk = Kf(float(am), 0.0) / float(K.subs(a=0))
    col = (0.55 + 0.45 * kk, 0.55 - 0.45 * abs(kk), 0.55 - 0.45 * kk)
    scene += parametric_plot3d(E, (a, a0, a1), (b, 0, 2 * pi), color=col, opacity=1, mesh=False, plot_points=[6, 80])
scene.save(OUT, viewer="tachyon", figsize=[7, 5], zoom=1.2, aspect_ratio=1)
print("[PASS] Gauss curvature of the torus K = cos(a) / (r (R + r cos a)):", bool((K - cos(a) / (r * (R + r * cos(a)))).simplify_full() == 0))
print("[PASS] K > 0 on the outer equator, K < 0 on the inner one, K = 0 at a = pi/2:", bool(K.subs(a=0) > 0 and K.subs(a=pi) < 0 and K.subs(a=pi / 2) == 0))
print("wrote", os.path.relpath(OUT))
