# Exercise exr-metrics-stereographic: the sphere in the stereographic chart from start to finish with SageManifolds:
# induced metric, conformal factor, angle preservation, the length of the equator and the area 4 pi.
E3 = Manifold(3, "R^3", structure="Riemannian"); cart = E3.chart("x y z")
delta = E3.metric("delta"); delta[0, 0] = 1; delta[1, 1] = 1; delta[2, 2] = 1
S = Manifold(2, "S", structure="Riemannian"); st = S.chart("u v"); u, v = st[:]
w = 1 + u**2 + v**2
Phi = S.diff_map(E3, {(st, cart): [2*u/w, 2*v/w, (u**2 + v**2 - 1)/w]})
g = S.metric("g"); g.set(Phi.pullback(delta))
Om2 = g[0, 0].expr().simplify_full()
print("conformal factor Omega^2 = g_uu =", Om2.factor(), "; g_uv =", g[0, 1].expr().simplify_full(), "; g_vv - g_uu =", (g[1, 1].expr() - g[0, 0].expr()).simplify_full())
X = S.vector_field(1, 2, name="X"); Y = S.vector_field(3, -1, name="Y")
cos_g = (g(X, Y) / sqrt(g(X, X) * g(Y, Y))).expr().simplify_full(); cos_e = (1*3 + 2*(-1)) / sqrt(5 * 10)
print("cos of the angle between X = (1, 2) and Y = (3, -1): on the sphere", cos_g, "; in the chart plane", cos_e, "; equal:", bool(cos_g == cos_e))
t = var("t", domain="real")
eq = S.curve({st: [cos(t), sin(t)]}, (t, 0, 2*pi)); vel = eq.tangent_vector_field()
speed = sqrt(g.along(eq)(vel, vel).expr()).simplify_full()
print("speed along the unit circle u^2 + v^2 = 1 (the equator):", speed, "; length:", integrate(speed, t, 0, 2*pi))
rho = var("rho", domain="real"); assume(rho > 0)
dens = Om2.subs(u=rho*cos(t), v=rho*sin(t)).simplify_full() * rho
print("area element in polar coordinates of the chart plane:", dens, "; area:", integrate(integrate(dens, t, 0, 2*pi), rho, 0, oo))
