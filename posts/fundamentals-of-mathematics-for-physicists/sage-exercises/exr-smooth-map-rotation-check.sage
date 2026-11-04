# Exercise exr-smooth-map-rotation-check: the rotation of S^2 about the Z-axis by alpha, written in the north
# stereographic chart as the plane rotation (u, v) -> (u cos alpha - v sin alpha, u sin alpha + v cos alpha) and defined
# as a SageManifolds diff_map: composed with the embedding it is the rotation R_z(alpha) of R^3, its Jacobian determinant
# is 1, and the rotation by -alpha is its inverse.
al = var("alpha", domain="real")
S2 = Manifold(2, "S^2"); U = S2.open_subset("U"); ste = U.chart("u v"); u, v = ste[:]
E3 = Manifold(3, "R^3"); C = E3.chart("X Y Z")
emb = U.diff_map(E3, {(ste, C): [2*u/(1 + u**2 + v**2), 2*v/(1 + u**2 + v**2), (u**2 + v**2 - 1)/(1 + u**2 + v**2)]}, name="E")
rot = U.diff_map(U, {(ste, ste): [u*cos(al) - v*sin(al), u*sin(al) + v*cos(al)]}, name="rot")
Rz = matrix(SR, [[cos(al), -sin(al), 0], [sin(al), cos(al), 0], [0, 0, 1]])
lhs = vector((emb * rot).expr(ste, C))                      # E after the rotation of the chart
rhs = Rz * vector(emb.expr(ste, C))                         # the rotation of R^3 after E
print("E o rot = R_z(alpha) o E:", all(bool(e.simplify_full() == 0) for e in lhs - rhs))
print("Jacobian determinant of rot:", jacobian(rot.expr(ste, ste), (u, v)).det().simplify_full())
inv = U.diff_map(U, {(ste, ste): [u*cos(al) + v*sin(al), -u*sin(al) + v*cos(al)]}, name="rot_inv")
print("rot(-alpha) o rot(alpha) is the identity of the chart:", [e.simplify_full() for e in (inv * rot).expr(ste, ste)])
print("the south pole (u, v) = (0, 0) is fixed:", rot.expr(ste, ste)[0].subs(u=0, v=0), rot.expr(ste, ste)[1].subs(u=0, v=0))
