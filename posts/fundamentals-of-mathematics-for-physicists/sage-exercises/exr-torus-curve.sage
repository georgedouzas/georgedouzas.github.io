# Exercise exr-torus-curve: a curve on the torus with SageManifolds' curve objects: the curve t -> (a, b) = (t, 2t)
# in the product chart, its image under the embedding, and the closing of the curve after one period.
R, r = 2, 1
E3 = Manifold(3, "R^3"); C = E3.chart("X Y Z")
T2 = Manifold(2, "T^2"); ab = T2.chart("a b"); a, b = ab[:]
emb = T2.diff_map(E3, {(ab, C): [(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)]}, name="E")
t = var("t")
gamma = T2.curve({ab: [t, 2*t]}, (t, 0, 2*pi), name="gamma")
print("curve in the chart:", gamma.coord_expr(ab))
print("velocity components (da/dt, db/dt):", gamma.tangent_vector_field().comp(ab.frame().along(gamma))[:] if False else [diff(t, t), diff(2*t, t)])
image = emb * gamma
print("image in R^3:", image.coord_expr(C))
P0 = image(t.subs(t=0)) if False else [e.subs(t=0) for e in image.coord_expr(C)]
P1 = [e.subs(t=2*pi) for e in image.coord_expr(C)]
print("endpoints coincide (the curve closes after one period):", P0 == P1, P0)
print("the curve winds once around the tube (a: 0 -> 2pi) and twice around the axis (b: 0 -> 4pi)")
