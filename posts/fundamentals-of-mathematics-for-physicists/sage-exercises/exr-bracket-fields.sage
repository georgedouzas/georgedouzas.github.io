# Exercise exr-bracket-fields: Lie brackets of vector fields on the sphere with SageManifolds' bracket(): the rotation
# field commutes with the gradient of the height, and [d_theta, sin(theta) d_phi] = cos(theta) d_phi.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
rot = A.vector_field(0, 1, name="rot")              # d/dphi
grad = A.vector_field(-sin(th), 0, name="gradZ")    # -sin(theta) d/dtheta
dth = A.vector_field(1, 0, name="dth")              # d/dtheta
Y = A.vector_field(0, sin(th), name="Y")            # sin(theta) d/dphi
print("[rot, gradZ] =", rot.bracket(grad).display())
print("[d_theta, sin(theta) d_phi] =", dth.bracket(Y).display())
f = A.scalar_field(function("f")(th, ph), name="f")
lhs = dth(Y(f)) - Y(dth(f))
print("the bracket acts on functions as X(Y f) - Y(X f):", bool((lhs - dth.bracket(Y)(f)).expr().simplify_full() == 0))
print("brackets are antisymmetric, [Y, d_theta] = -[d_theta, Y]:", Y.bracket(dth) == -dth.bracket(Y))
