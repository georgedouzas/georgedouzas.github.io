# Exercise exr-stereographic-gram: the Gram matrix of the inverse stereographic chart, computed with SageManifolds
# as the pullback of the Euclidean metric of R^3, and the conformal factor 4/(1 + u^2 + v^2)^2.
E3 = Manifold(3, "R^3", structure="Riemannian"); C = E3.chart("X Y Z")
g3 = E3.metric(); g3[0, 0] = 1; g3[1, 1] = 1; g3[2, 2] = 1
S2 = Manifold(2, "S^2", structure="Riemannian"); ste = S2.chart("u v"); u, v = ste[:]
emb = S2.diff_map(E3, {(ste, C): [2*u/(1 + u**2 + v**2), 2*v/(1 + u**2 + v**2), (u**2 + v**2 - 1)/(1 + u**2 + v**2)]}, name="E")
G = S2.metric(); G.set(emb.pullback(g3))
print("Gram matrix of the stereographic parametrisation:"); print(G.display())
print("G_uu = G_vv = 4/(1 + u^2 + v^2)^2:", bool((G[0, 0].expr() - 4/(1 + u**2 + v**2)**2).simplify_full() == 0) and bool(G[0, 1].expr() == 0))
print("determinant:", G.determinant().expr().factor(), " -> never zero: the chart is regular everywhere")
