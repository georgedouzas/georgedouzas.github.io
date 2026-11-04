# Exercise exr-curve-length: the length of a curve from the metric, with SageManifolds: a parallel and a slanted curve
# on the unit sphere, compared with the straight-line distance in the chart.
S2 = Manifold(2, "S^2", structure="Riemannian"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
g = A.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th)**2
t = var("t", domain="real")
def length(curve, t0, t1):
    v = curve.tangent_vector_field()
    speed = sqrt(g.along(curve)(v, v).expr()).simplify_full()
    return integrate(speed, t, t0, t1)
par = A.curve({sph: [pi/3, t]}, (t, 0, 2*pi), name="parallel")
print("length of the parallel theta = pi/3:", length(par, 0, 2*pi), " = 2 pi sin(pi/3)")
mer = A.curve({sph: [t, 1]}, (t, 0.01, pi - 0.01), name="meridian")
print("length of a meridian from theta = 0.01 to pi - 0.01:", length(mer, 0.01, pi - 0.01).n(digits=6), " -> pi in the limit")
sl = A.curve({sph: [pi/2 + 0.3*t, t]}, (t, 0, 1), name="slanted")
print("length of the slanted curve (theta, phi) = (pi/2 + 0.3 t, t), t in [0, 1]:", length(sl, 0, 1).n(digits=6), "; chart distance sqrt(0.3^2 + 1) =", sqrt(0.09 + 1).n(digits=6))
