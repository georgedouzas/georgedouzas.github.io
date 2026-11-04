# Exercise exr-stereographic-transition: the transition map between the two stereographic charts of S^2, built with
# SageManifolds: the sphere as a 2-manifold with two charts and the transition map between them.
S2 = Manifold(2, "S^2", structure="topological")
U = S2.open_subset("U"); V = S2.open_subset("V")      # sphere minus the north pole, minus the south pole
S2.declare_union(U, V)
N = U.chart("u v")                                    # projection from the north pole
S = V.chart("up:u' vp:v'")                            # projection from the south pole
u, v = N[:]; up, vp = S[:]
W = U.intersection(V)
tr = N.transition_map(S, (u / (u**2 + v**2), v / (u**2 + v**2)), intersection_name="W", restrictions1=u**2 + v**2 != 0, restrictions2=up**2 + vp**2 != 0)
tr.set_inverse(up / (up**2 + vp**2), vp / (up**2 + vp**2))
print("transition map:", tr.display())
print("its inverse is the same formula:", tr.inverse().display())
J = jacobian(tr(u, v), (u, v))
print("Jacobian determinant:", J.det().factor())
# the north-chart point lies on the sphere
X = 2*u/(1 + u**2 + v**2); Y = 2*v/(1 + u**2 + v**2); Z = (u**2 + v**2 - 1)/(1 + u**2 + v**2)
print("the inverse north chart lands on the sphere, X^2 + Y^2 + Z^2 = 1:", bool((X**2 + Y**2 + Z**2).simplify_full() == 1))
print("and the south-chart coordinates of that point, (X, Y)/(1 + Z), are the inversion:", bool((X/(1 + Z)).simplify_full() == u/(u**2 + v**2)))
