# Exercise exr-lg-haar-measure: the wedge product of the left-invariant forms is a left-invariant volume form (a Haar
# measure). On SU(2), sigma^1 ^ sigma^2 ^ sigma^3 = sin(theta) dtheta ^ dphi ^ dpsi with total volume 16 pi^2; the metric
# (1/4) sum sigma^a sigma^a has volume form (1/8) sigma^1 ^ sigma^2 ^ sigma^3 and volume 2 pi^2, that of the unit S^3.
G = Manifold(3, "SU2"); X.<th, ph, ps> = G.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi")
s1 = G.one_form(cos(ps), sin(th) * sin(ps), 0); s2 = G.one_form(-sin(ps), sin(th) * cos(ps), 0); s3 = G.one_form(0, cos(th), 1)
vol = s1.wedge(s2).wedge(s3)
print("sigma^1 ^ sigma^2 ^ sigma^3 =", vol.display())
dens = vol[0, 1, 2].expr().simplify_full()
V = integrate(integrate(integrate(dens, th, 0, pi), ph, 0, 2 * pi), ps, 0, 4 * pi)
print("total volume =", V, "; with the metric's volume form (1/8) of it:", V / 8, "= 2 pi^2:", bool(V / 8 == 2 * pi^2))
g = G.metric("g"); g[0, 0], g[1, 1], g[2, 2], g[1, 2] = 1/4, 1/4, 1/4, cos(th) / 4
print("sqrt(det g) =", sqrt(g.determinant().expr()).simplify_full(), "(= sin(theta)/8 on 0 < theta < pi)")
A = Manifold(2, "Aff"); Y.<a, b> = A.chart("a:(0,+oo) b")
v = A.one_form(1 / a, 0).wedge(A.one_form(0, 1 / a))
var("a0 b0", domain="real")
Lh = A.diff_map(A, [a0 * a, a0 * b + b0])
print("affine group: sigma^X ^ sigma^Y =", v.display(), "; left-invariant:", Lh.pullback(v) == v)
