# Exercise exr-curve-length-torus: lengths of curves on the embedded torus (R = 2, r = 4/5) from the induced metric
# with SageManifolds: a tube circle, the outer and inner axis circles and the diagonal curve a = b = t.
R, r = 2, QQ(4)/5
T = Manifold(2, "T^2", structure="Riemannian"); X = T.chart("a:(0,2*pi) b:(0,2*pi)"); a, b = X[:]
g = T.metric("g"); g[0, 0] = r**2; g[1, 1] = (R + r*cos(a))**2
t = var("t", domain="real")
def length(curve, t0, t1, numeric=False):
    v = curve.tangent_vector_field()
    speed = sqrt(g.along(curve)(v, v).expr()).simplify_full()
    return numerical_integral(speed, t0, t1)[0] if numeric else integrate(speed, t, t0, t1)
tube = T.curve({X: [t, 1]}, (t, 0, 2*pi)); print("tube circle b = 1:", length(tube, 0, 2*pi), "= 2 pi r")
outer = T.curve({X: [0, t]}, (t, 0, 2*pi)); inner = T.curve({X: [pi, t]}, (t, 0, 2*pi))
print("outer axis circle a = 0:", length(outer, 0, 2*pi), "= 2 pi (R + r); inner axis circle a = pi:", length(inner, 0, 2*pi), "= 2 pi (R - r)")
diag = T.curve({X: [t, t]}, (t, 0, 2*pi))
print("diagonal curve a = b = t, t in [0, 2 pi]: length", length(diag, 0, 2*pi, numeric=True), "; chart length 2 pi sqrt(2) =", (2*pi*sqrt(2)).n(digits=6))
