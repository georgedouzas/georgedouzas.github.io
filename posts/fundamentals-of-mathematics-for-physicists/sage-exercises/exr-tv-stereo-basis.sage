# Exercise exr-tv-stereo-basis: the stereographic basis vectors d_u, d_v as ambient vectors, through the Jacobian of
# the embedding E(u, v) = (2u, 2v, u^2 + v^2 - 1)/(1 + u^2 + v^2): tangent to the sphere, orthogonal, of equal length.
P = Manifold(2, "U"); ste = P.chart("u v"); u, v = ste[:]
E3 = Manifold(3, "R^3"); C = E3.chart("X Y Z")
E = P.diff_map(E3, {(ste, C): [2*u/(1 + u**2 + v**2), 2*v/(1 + u**2 + v**2), (u**2 + v**2 - 1)/(1 + u**2 + v**2)]}, name="E")
J = E.jacobian_matrix(ste, C)
Eu, Ev = J.column(0), J.column(1)
pos = vector(E.expr(ste, C))
print("E lies on the unit sphere:", bool((pos*pos).simplify_full() == 1))
print("d_u E and d_v E are tangent (orthogonal to the position vector):", bool((Eu*pos).simplify_full() == 0 and (Ev*pos).simplify_full() == 0))
print("d_u E . d_v E = 0:", bool((Eu*Ev).simplify_full() == 0))
print("|d_u E|^2 = |d_v E|^2 =", (Eu*Eu).factor(), ":", bool((Eu*Eu - Ev*Ev).simplify_full() == 0))
print("length at the origin (south pole), on the unit circle (equator), at u = 3:", [sqrt((Eu*Eu).subs(u=a, v=0)) for a in [0, 1, 3]])
