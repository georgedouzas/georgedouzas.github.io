# Exercise exr-volume-torus: the area of the embedded torus from the volume element sqrt(det g) of the induced
# metric, integrated over the chart with SageManifolds, compared with the flat torus of the same radii.
R, r = var("R r", domain="real"); assume(R > r); assume(r > 0)
T = Manifold(2, "T^2", structure="Riemannian"); X = T.chart("a:(0,2*pi) b:(0,2*pi)"); a, b = X[:]
g = T.metric("g"); g[0, 0] = r**2; g[1, 1] = (R + r*cos(a))**2
vol = r*(R + r*cos(a))
print("det g =", g.determinant().expr().factor(), "; sqrt(det g) = r (R + r cos a):", bool((vol**2 - g.determinant().expr()).simplify_full() == 0))
area = integrate(integrate(vol, b, 0, 2*pi), a, 0, 2*pi)
print("area of the embedded torus:", area.factor())
flat = T.metric("g_flat"); flat[0, 0] = r**2; flat[1, 1] = R**2
vflat = sqrt(flat.determinant().expr()).simplify_full()
print("volume element of the flat torus r^2 da^2 + R^2 db^2:", vflat, "; area:", integrate(integrate(vflat, b, 0, 2*pi), a, 0, 2*pi))
print("the two areas agree:", bool(area == 4*pi**2*R*r))
