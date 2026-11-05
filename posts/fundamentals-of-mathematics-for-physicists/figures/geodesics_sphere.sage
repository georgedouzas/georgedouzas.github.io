# Figure: geodesics of the round sphere, integrated numerically from the geodesic equation with SageManifolds'
# integrated_geodesic and drawn on the embedded sphere: from one point, geodesics leaving in several directions are
# great circles (blue); the parallel theta = pi/3 (red) is not a geodesic, it bends away from the great circle
# with the same initial velocity. Output: figures/geodesics_sphere.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geodesics_sphere.png")
S2 = Manifold(2, "S^2", structure="Riemannian"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
g = A.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th)**2
E3 = Manifold(3, "R^3"); C = E3.chart("X Y Z")
emb = A.diff_map(E3, {(sph, C): [sin(th)*cos(ph), sin(th)*sin(ph), cos(th)]}, name="E")
t = var("t")
ph0 = 5.2
p0 = A.point((pi / 3, ph0), name="p")
scene = sphere((0, 0, 0), 0.985, color="whitesmoke", opacity=0.95)
scene += sum(sphere((float(sin(pi/3)*cos(q)), float(sin(pi/3)*sin(q)), float(cos(pi/3))), 0.016, color=(0.7, 0.13, 0.13)) for q in srange(ph0 - 1.8 / sin(pi/3), ph0 + 0.001, 0.04))
scene += point3d((sin(pi/3)*cos(ph0), sin(pi/3)*sin(ph0), cos(pi/3)), size=15, color="black")
Tp = A.tangent_space(p0)
for (vth, vph) in [(0, -1 / sin(pi / 3)), (1, 0), (0.7, -0.7 / sin(pi / 3)), (-0.7, -0.7 / sin(pi / 3))]:
    v = Tp((vth, vph))
    geo = A.integrated_geodesic(g, (t, 0, 1.8), v, name="geo")           # unit speed; stays inside the chart
    sol = geo.solve(step=0.02, verbose=False)                               # list of [t, theta, phi]
    pts = [(float(sin(q[1]) * cos(q[2])), float(sin(q[1]) * sin(q[2])), float(cos(q[1]))) for q in sol]
    scene += sum(sphere(q, 0.016, color=(0.25, 0.41, 0.88)) for q in pts[::2])
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.3, aspect_ratio=1)
# geodesic equation residual for the equator and a meridian, symbolically
Gam = g.christoffel_symbols()
eq_th = 0 + Gam[0, 1, 1].expr().subs(th=pi / 2) * 1 * 1          # theta'' + Gamma^th_phph phi'^2 with theta = pi/2, phi = t
mer = (0 + Gam[1, 0, 0].expr(), 0 + Gam[0, 0, 0].expr())         # meridian theta = t: residuals involve Gamma^k_thth only
print("[PASS] the equator theta = pi/2, phi = t satisfies the geodesic equation:", bool(eq_th == 0))
print("[PASS] meridians theta = t, phi = const satisfy it (Gamma^k_thth = 0):", bool(mer[0] == 0 and mer[1] == 0))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the same sphere in the threejs viewer, with the geodesics (blue), the parallel
# theta = pi/3 (red) and, along the whole parallel, the vector d/dtheta parallel-transported once around it (green
# arrows, every 30 degrees): it turns relative to the parallel and comes back rotated by 2 pi cos(pi/3) = pi, so the
# holonomy angle 2 pi (1 - cos(pi/3)) = pi equals the area of the polar cap above the parallel.
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
_th0 = pi / 3
_c, _s = float(cos(_th0)), float(sin(_th0))
_iscene = sphere((0, 0, 0), 0.985, color="whitesmoke", opacity=0.6)
_iscene += parametric_plot3d((_s * cos(t), _s * sin(t), _c), (t, 0, 2 * pi), color="firebrick", thickness=3)
for (vth, vph) in [(0, -1 / sin(pi / 3)), (1, 0), (0.7, -0.7 / sin(pi / 3)), (-0.7, -0.7 / sin(pi / 3))]:
    _geo = A.integrated_geodesic(g, (t, 0, 1.8), Tp((vth, vph)), name="geo"); _sol = _geo.solve(step=0.02, verbose=False)
    _iscene += line3d([(float(sin(q[1]) * cos(q[2])), float(sin(q[1]) * sin(q[2])), float(cos(q[1]))) for q in _sol], color="royalblue", thickness=3)
def _transported(tt):
    """d/dtheta transported along theta = pi/3 to phi = tt (exact solution of the transport equation), in R^3"""
    vth, vph = cos(tt * _c), -sin(tt * _c) / _s
    e_th = vector(RDF, [_c * cos(tt), _c * sin(tt), -_s]); e_ph = vector(RDF, [-_s * sin(tt), _s * cos(tt), 0])
    return float(vth) * e_th + float(vph) * e_ph
for k in range(13):
    tt = 2 * pi * k / 12
    P = vector(RDF, [_s * cos(tt), _s * sin(tt), _c])
    _iscene += arrow3d(tuple(P), tuple(P + 0.25 * _transported(tt)), color="darkgreen" if k < 12 else "black", width=2)
threejs_html("geodesics_sphere", _iscene, frame=False)
_V0, _V1 = _transported(0), _transported(2 * pi)
print("[PASS] the transported vector keeps unit length around the parallel:", all(abs(_transported(2 * pi * k / 24).norm() - 1) < 1e-12 for k in range(25)))
print("[PASS] after one turn around theta = pi/3 the transported d/dtheta is reversed (rotated by pi = cap area):", bool(abs(_V0.dot_product(_V1) + 1) < 1e-12))
print("wrote figures/interactive/geodesics_sphere.html")
