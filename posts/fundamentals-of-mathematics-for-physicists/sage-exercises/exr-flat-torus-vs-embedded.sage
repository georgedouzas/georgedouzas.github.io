# Exercise exr-flat-torus-vs-embedded: the flat torus r1^2 da^2 + r2^2 db^2 has vanishing Christoffel symbols and
# Riemann tensor, while the embedded torus r^2 da^2 + (R + r cos a)^2 db^2 is curved (SageManifolds).
r1, r2 = var("r1 r2", domain="real"); assume(r1 > 0); assume(r2 > 0)
T = Manifold(2, "T^2", structure="Riemannian"); X = T.chart("a:(0,2*pi) b:(0,2*pi)"); a, b = X[:]
flat = T.metric("g_flat"); flat[0, 0] = r1**2; flat[1, 1] = r2**2
print("flat torus:", flat.display())
print("Christoffel symbols all vanish:", all(flat.christoffel_symbols()[k, i, j].expr() == 0 for k in range(2) for i in range(2) for j in range(2)))
print("Riemann tensor vanishes:", flat.riemann() == 0, "; area =", integrate(integrate(sqrt(flat.determinant().expr()).simplify_full(), b, 0, 2*pi), a, 0, 2*pi))
R, r = 2, QQ(4)/5
emb = T.metric("g_emb"); emb[0, 0] = r**2; emb[1, 1] = (R + r*cos(a))**2
print("embedded torus R = 2, r = 4/5:", emb.display())
print("Riemann tensor vanishes:", emb.riemann() == 0, "; component R^a_{bab} =", emb.riemann()[0, 1, 0, 1].expr().simplify_full())
print("scalar curvature:", emb.ricci_scalar().expr().simplify_full(), " (positive on the outer equator a = 0, negative on the inner one a = pi)")
