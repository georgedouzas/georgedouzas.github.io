# Exercise exr-raising-gradient-polar: gradients in polar coordinates with SageManifolds: raising the index of df with
# the inverse polar metric for f = r^2, f = phi and f = r cos(phi) = x, whose gradient is the unit field d_x.
M = Manifold(2, "R^2", structure="Riemannian"); U = M.open_subset("U")
pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
g = U.metric("g"); g[0, 0] = 1; g[1, 1] = r**2
for name, expr in [("r^2", r**2), ("phi", ph), ("x = r cos(phi)", r*cos(ph))]:
    f = U.scalar_field(expr, name="f"); df = f.differential(); grad = df.up(g)
    print("f =", name, ": df =", df.display(), "; grad f =", grad.display(), "; |grad f| =", sqrt(g(grad, grad).expr()).simplify_full())
