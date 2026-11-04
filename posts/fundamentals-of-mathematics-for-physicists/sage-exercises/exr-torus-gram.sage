# Exercise exr-torus-gram: the Gram matrix of the standard embedding of the torus T^2 in R^3, computed with
# SageManifolds as the induced (pullback) metric of the Euclidean metric under the embedding, and by hand as J^T J.
R, r = var("R r", domain="real"); assume(R > r, r > 0)
E3 = Manifold(3, "R^3", structure="Riemannian")
C = E3.chart("X Y Z"); g3 = E3.metric(); g3[0, 0] = 1; g3[1, 1] = 1; g3[2, 2] = 1
T2 = Manifold(2, "T^2", structure="Riemannian")
ab = T2.chart("a:(0,2*pi) b:(0,2*pi)"); a, b = ab[:]
emb = T2.diff_map(E3, {(ab, C): [(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)]}, name="E")
G = T2.metric(); G.set(emb.pullback(g3))
print("induced metric (Gram matrix):"); print(G.display())
print("determinant:", G.determinant().expr().factor())
J = jacobian([(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)], (a, b))
Gm = matrix([[G[i, j].expr() for j in range(2)] for i in range(2)])
print("J^T J agrees with the pullback metric:", ((J.transpose()*J) - Gm).simplify_full() == 0)
