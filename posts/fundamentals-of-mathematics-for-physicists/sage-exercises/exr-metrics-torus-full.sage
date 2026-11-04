# Exercise exr-metrics-torus-full: the embedded torus from start to finish with SageManifolds: induced metric by pullback,
# signature, determinant, inverse, the lowered rotation field, the gradient of the height, an angle and the area.
R, r = 2, QQ(4)/5
E3 = Manifold(3, "R^3", structure="Riemannian"); cart = E3.chart("x y z")
delta = E3.metric("delta"); delta[0, 0] = 1; delta[1, 1] = 1; delta[2, 2] = 1
T = Manifold(2, "T^2", structure="Riemannian"); X = T.chart("a:(0,2*pi) b:(0,2*pi)"); a, b = X[:]
Phi = T.diff_map(E3, {(X, cart): [(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)]})
g = T.metric("g"); g.set(Phi.pullback(delta))
print("induced metric:", g.display())
G0 = matrix(QQ, [[g[i, j].expr().subs(a=0, b=0) for j in range(2)] for i in range(2)])
print("at the outer equator a = 0 the matrix is", G0.list(), "with signature (p, q, z) =", QuadraticForm(QQ, 2*G0).signature_vector())
print("determinant:", g.determinant().expr().factor(), "; inverse:", g.inverse().display())
db = T.vector_field(0, 1, name="db"); print("lowering d_b:", db.down(g).display())
z = T.scalar_field(r*sin(a), name="z"); grad = z.differential().up(g)
print("gradient of the height z = r sin a:", grad.display(), "; |grad z| =", sqrt(g(grad, grad).expr()).simplify_full())
V = T.vector_field(1, 1, name="V")
c = (g(V, db) / sqrt(g(V, V) * g(db, db))).expr()
print("cos of the angle between d_a + d_b and d_b at the outer equator:", c.subs(a=0).simplify_full(), "=", c.subs(a=0).n(digits=5), "; on the inner equator:", c.subs(a=pi).simplify_full())
area = integrate(integrate(r*(R + r*cos(a)), b, 0, 2*pi), a, 0, 2*pi)
print("sqrt(det g) = r (R + r cos a):", bool((g.determinant().expr() - r**2*(R + r*cos(a))**2).simplify_full() == 0), "; area 4 pi^2 R r =", area)
