# Exercise exr-tv-rotation-stereographic: the three rotation fields K1, K2, K3 of the sphere in the north stereographic
# chart, with SageManifolds: K1 = uv d_u + (1 - u^2 + v^2)/2 d_v, K2 = -(1 + u^2 - v^2)/2 d_u - uv d_v, K3 = -v d_u + u d_v,
# polynomial components with no singularity at the south pole u = v = 0.
S2 = Manifold(2, "S^2")
A = S2.open_subset("A"); U = S2.open_subset("U")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); ste = U.chart("u v")
th, ph = sph[:]; u, v = ste[:]
tr = sph.transition_map(ste, (sin(th)*cos(ph)/(1 - cos(th)), sin(th)*sin(ph)/(1 - cos(th))))
tr.set_inverse(2*atan(1/sqrt(u**2 + v**2)), atan2(v, u), check=False)
W = A.intersection(U); fs = sph.frame().restrict(W); fu = ste.frame().restrict(W)
U_sph, V_sph = tr(th, ph)                                  # u and v as functions of (theta, phi)
K = [W.vector_field(name="K%d" % a) for a in (1, 2, 3)]
K[0][fs, :] = [-sin(ph), -cos(th)/sin(th)*cos(ph)]
K[1][fs, :] = [cos(ph), -cos(th)/sin(th)*sin(ph)]
K[2][fs, :] = [0, 1]
expected = [[u*v, (1 - u**2 + v**2)/2], [-(1 + u**2 - v**2)/2, -u*v], [-v, u]]
for a in range(3):
    print(K[a].display(fu, ste.restrict(W)))
    c = [ci.expr() for ci in K[a][fu, :, ste.restrict(W)]]
    print("   components equal", expected[a], ":", all(bool((c[i] - expected[a][i]).simplify_full() == 0) for i in range(2)))
