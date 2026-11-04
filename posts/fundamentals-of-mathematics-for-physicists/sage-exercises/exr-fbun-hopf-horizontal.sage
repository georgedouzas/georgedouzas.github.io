# Exercise exr-fbun-hopf-horizontal: on SU(2) in Euler angles (theta, phi, psi) the Hopf connection
# sigma^3 = d psi + cos(theta) d phi is 1 on the fibre generator d/dpsi, is invariant under the flow of d/dpsi
# (Lie derivative zero), and its kernel, the horizontal plane, is spanned by d/dtheta and d/dphi - cos(theta) d/dpsi.
G = Manifold(3, "SU2"); U = G.open_subset("U")
E = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi"); th, ph, ps = E[:]
s3 = U.one_form(0, cos(th), 1, name="sigma3")
V = U.vector_field(0, 0, 1, name="V")                      # generator of the right U(1) action
H1 = U.vector_field(1, 0, 0, name="H1")
H2 = U.vector_field(0, 1, -cos(th), name="H2")
print("sigma3(d/dpsi) =", s3(V).expr())
print("Lie derivative of sigma3 along d/dpsi:", s3.lie_der(V).display())
print("sigma3(H1) =", s3(H1).expr(), "; sigma3(H2) =", s3(H2).expr())
M = matrix([[1, 0, 0], [0, 1, -cos(th)], [0, 0, 1]])
print("H1, H2, d/dpsi are a basis (determinant", M.det(), "): horizontal plus vertical span the tangent space")
print("the horizontal fields project to d/dtheta and d/dphi on the base (their first two components):",
      [H1[i].expr() for i in range(2)], [H2[i].expr() for i in range(2)])
