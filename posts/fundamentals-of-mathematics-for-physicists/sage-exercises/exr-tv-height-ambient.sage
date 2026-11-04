# Exercise exr-tv-height-ambient: the tangent vector X = d_theta + 2 d_phi acting on the three ambient coordinate
# functions of the sphere: the three numbers are the components of the ambient vector X^i d_i E.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
E = vector([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)])
X = A.vector_field(1, 2, name="X")
vals = [X(A.scalar_field(E[k])).expr() for k in range(3)]
print("(X(X_amb), X(Y_amb), X(Z)) =", vals)
amb = E.diff(th) + 2*E.diff(ph)
print("equal to the ambient vector d_theta E + 2 d_phi E:", bool(all((vals[k] - amb[k]).simplify_full() == 0 for k in range(3))))
print("X(Z) = -sin(theta) (the proposition):", bool(vals[2] == -sin(th)))
print("at (theta, phi) = (pi/3, pi/4):", [ex.subs(th=pi/3, ph=pi/4).simplify_full() for ex in vals])
