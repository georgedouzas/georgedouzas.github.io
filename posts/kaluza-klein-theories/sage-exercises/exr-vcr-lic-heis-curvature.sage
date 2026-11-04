# Exercise exr-vcr-lic-heis-curvature: the left-invariant metric dy^2 + dz2^2 + (dz1 + m y dz2)^2 on the Heisenberg
# nilmanifold. Its frame Ricci components are constants and the scalar curvature is R = -m^2/2 < 0, so the potential
# -R of the reduction is positive and has no critical point in m: the twist only pushes the volume.
N = Manifold(3, "Nil"); X = N.chart("y z1 z2"); y, z1, z2 = X[:]
m = var("m", domain="positive")
S = matrix(SR, [[1, 0, 0], [0, 1, m * y], [0, 0, 1]])        # rows: sigma^y, sigma^1, sigma^2
g = N.metric("g"); g[:] = (S.transpose() * S).simplify_full()
print("metric:"); print(g[:])
print("scalar curvature R =", g.ricci_scalar().expr().simplify_full())
Sinv = S.inverse()
E = [N.vector_field(*[Sinv[i, k] for i in range(3)]) for k in range(3)]
Ric = g.ricci()
print("frame Ricci matrix (y, 1, 2):"); print(matrix(SR, 3, 3, lambda i, j: Ric(E[i], E[j]).expr().simplify_full()))
