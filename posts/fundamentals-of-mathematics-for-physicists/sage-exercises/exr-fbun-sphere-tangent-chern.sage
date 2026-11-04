# Exercise exr-fbun-sphere-tangent-chern: the Levi-Civita connection of the round sphere in the orthonormal frame
# e_1 = d/dtheta, e_2 = (1/sin theta) d/dphi has the single connection one-form omega^1_2 = -cos(theta) d phi; its
# curvature d omega^1_2 = sin(theta) d theta ^ d phi integrates to 4 pi, so the tangent bundle, as a U(1) bundle,
# has |c_1| = 2 = Euler number of S^2, the winding number of its transition function between stereographic charts.
U = Manifold(2, "S2", structure="Riemannian")          # the sphere minus a meridian, one chart
sph = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
g = U.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th) ** 2
e = U.vector_frame("e", (U.vector_field(1, 0), U.vector_field(0, 1 / sin(th))))
nab = g.connection()
w = nab.connection_form(0, 1, e)
print("omega^1_2 =", w.display(sph.frame()))
dw = w.exterior_derivative()
print("d omega^1_2 =", dw.display(sph.frame()))
print("Gauss curvature K = R_1212:", g.riemann()[0, 1, 0, 1].expr() / sin(th) ** 2)
tot = integrate(integrate(dw[sph.frame(), 0, 1].expr(), ph, 0, 2 * pi), th, 0, pi)
print("int d omega^1_2 =", tot, "; (1/2 pi) int =", tot / (2 * pi))
print("compare with the Hopf connection: omega^1_2 = -cos(theta) d phi is minus the potential A = cos(theta) d phi")
