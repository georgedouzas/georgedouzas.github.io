# Exercise exr-induced-metric-cylinder-cone: induced metrics of the cylinder and the cone as pullbacks of the Euclidean
# metric of R^3 with SageManifolds; both have vanishing Riemann tensor although the cone's components are not constant.
E3 = Manifold(3, "R^3", structure="Riemannian"); cart = E3.chart("x y z")
delta = E3.metric("delta"); delta[0, 0] = 1; delta[1, 1] = 1; delta[2, 2] = 1
R = var("R", domain="real"); assume(R > 0)
C = Manifold(2, "Cyl", structure="Riemannian"); Xc = C.chart(r"u ph:(0,2*pi):\varphi"); u, ph = Xc[:]
Phi = C.diff_map(E3, {(Xc, cart): [R*cos(ph), R*sin(ph), u]})
gc = C.metric("g"); gc.set(Phi.pullback(delta))
print("cylinder:", gc.display(), "; det =", gc.determinant().expr().simplify_full(), "; Riemann = 0:", gc.riemann() == 0)
K = Manifold(2, "Cone", structure="Riemannian"); Xk = K.chart(r"s:(0,+oo) ph:(0,2*pi):\varphi"); s, phk = Xk[:]
Psi = K.diff_map(E3, {(Xk, cart): [s*cos(phk), s*sin(phk), s]})
gk = K.metric("g"); gk.set(Psi.pullback(delta))
print("cone z = sqrt(x^2 + y^2):", gk.display(), "; det =", gk.determinant().expr().simplify_full(), "; Riemann = 0:", gk.riemann() == 0)
print("the cone's circle s = 1 has length", integrate(sqrt(gk[1, 1].expr()).subs(s=1), phk, 0, 2*pi), "while its radius, measured on the cone, is", integrate(sqrt(gk[0, 0].expr()), s, 0, 1))
