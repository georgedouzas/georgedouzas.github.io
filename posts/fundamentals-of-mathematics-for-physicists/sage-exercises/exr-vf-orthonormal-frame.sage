# Exercise exr-vf-orthonormal-frame: the frame e1 = d_theta, e2 = d_phi/sin(theta) of the sphere is orthonormal for
# the ambient dot product and is not a coordinate frame, [e1, e2] = -cot(theta) e2.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
E = vector([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)])
J = jacobian(E, (th, ph))
e1 = A.vector_field(1, 0, name="e1"); e2 = A.vector_field(0, 1/sin(th), name="e2")
a1 = J*vector([1, 0]); a2 = J*vector([0, 1/sin(th)])
print("|e1|^2, |e2|^2, e1.e2 in R^3:", (a1*a1).simplify_full(), (a2*a2).simplify_full(), (a1*a2).simplify_full())
print("[e1, e2] =", e1.bracket(e2).display())
print("[e1, e2] = -cot(theta) e2:", e1.bracket(e2) == -(cos(th)/sin(th))*e2)
f = A.scalar_field(function("f")(th, ph), name="f")
print("check on a function, e1(e2 f) - e2(e1 f) = -cot(theta) e2(f):", bool((e1(e2(f)) - e2(e1(f)) + cos(th)/sin(th)*e2(f)).expr().simplify_full() == 0))
