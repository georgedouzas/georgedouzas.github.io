# Exercise exr-vf-rotation-fields: the three rotation fields K1, K2, K3 of the sphere, pushed to R^3 through the
# Jacobian of the embedding: K_a is the ambient vector field e_a x p, tangent to the sphere and vanishing at +-e_a.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
K1 = A.vector_field(-sin(ph), -cos(th)/sin(th)*cos(ph), name="K1")
K2 = A.vector_field(cos(ph), -cos(th)/sin(th)*sin(ph), name="K2")
K3 = A.vector_field(0, 1, name="K3")
K = [K1, K2, K3]
E = vector([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)])
J = jacobian(E, (th, ph))
e = [vector([1, 0, 0]), vector([0, 1, 0]), vector([0, 0, 1])]
for a in range(3):
    amb = J * vector([K[a][i].expr() for i in range(2)])
    print("K%d as an ambient vector equals e_%d x p:" % (a + 1, a + 1), bool(all((amb[k] - e[a].cross_product(E)[k]).simplify_full() == 0 for k in range(3))))
    print("   it is tangent, (K%d . p) = 0:" % (a + 1), bool((amb*E).simplify_full() == 0), "; its squared length:", (amb*amb).simplify_full())
