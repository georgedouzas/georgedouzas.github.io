# Exercise exr-flows-killing-sphere-isometries: the flow of the Killing field K_1 of the sphere is the rotation about
# the X-axis, a one-parameter group of isometries. Checked with SageManifolds: the rotated embedding Phi_t = R_X(t) o E
# has generator J K_1, satisfies the group law, and pulls the Euclidean metric of R^3 back to the round metric for
# every t, while the flow of the conformal field -sin(theta) d/dtheta does not.
t, s = var("t s", domain="real")
S2 = Manifold(2, "S2", structure="Riemannian"); C = S2.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = C[:]
g = S2.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th) ** 2
R3 = EuclideanSpace(3, "R3"); cart = R3.cartesian_coordinates(); delta = R3.metric()
E = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
RX = lambda a: matrix([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])
Phi = lambda a: RX(a) * E
K1 = vector([-sin(ph), -cos(th) / sin(th) * cos(ph)])
gen = diff(Phi(t), t).subs(t=0); JK1 = jacobian(E, (th, ph)) * K1
print("generator d/dt|0 Phi_t = J K_1 (the ambient field xi_1 = (0, -z, y)):", bool(all((a - b).simplify_full() == 0 for a, b in zip(gen, JK1))))
print("group law Phi_s o Phi_t = Phi_{s+t} (R_X(s) R_X(t) = R_X(s+t)):", bool(all((a - b).simplify_trig() == 0 for a, b in zip((RX(s) * RX(t)).list(), RX(s + t).list()))))
F = S2.diff_map(R3, list(Phi(t)), name="Phi_t")
h = F.pullback(delta); h.apply_map(lambda e: e.simplify_full())
print("Phi_t^* delta =", h.display(), " = g for every t:", h == g)
# the flow of the conformal field X = -sin(th) d/dth: theta -> 2 arctan(tan(theta/2) e^{-t}), phi fixed
G = S2.diff_map(S2, [2 * arctan(tan(th / 2) * exp(-t)), ph], name="Psi_t")
k = G.pullback(g); k.apply_map(lambda e: e.simplify_full())
print("conformal flow: Psi_t^* g == g:", k == g, ";  ratio of the theta-theta components:", (k[0, 0] / g[0, 0]).expr().simplify_full().factor())
