# Exercise exr-vcr-lic-berger-sage: the Berger (squashed) three-sphere a^2 (sigma_1^2 + sigma_2^2) + b^2 sigma_3^2 with
# the left-invariant forms of SU(2) in Euler angles: scalar curvature and the frame components of the Ricci tensor.
M = Manifold(3, "SU2"); X = M.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi"); th, ph, ps = X[:]
S = matrix(SR, [[cos(ps), sin(th) * sin(ps), 0], [-sin(ps), sin(th) * cos(ps), 0], [0, cos(th), 1]])   # rows: sigma_a
a, b = var("a b", domain="positive")
G = (S.transpose() * diagonal_matrix([a ^ 2, a ^ 2, b ^ 2]) * S).simplify_full()
g = M.metric("g"); g[:] = G
R = g.ricci_scalar().expr().simplify_full()
print("R =", R.factor(), "; equals (4a^2 - b^2)/(2a^4):", bool((R - (4 * a ^ 2 - b ^ 2) / (2 * a ^ 4)).simplify_full() == 0))
Sinv = S.inverse().simplify_full()
E = [M.vector_field(*[Sinv[i, k] for i in range(3)]) for k in range(3)]
Ric = g.ricci()
print("frame Ricci matrix Ric(e_a, e_b):"); print(matrix(SR, 3, 3, lambda i, j: Ric(E[i], E[j]).expr().simplify_full()))
print("round sphere a = b = 1/2 (radius 1): R =", R.subs(a=1 / 2, b=1 / 2), "; R < 0 once b > 2a, e.g. b = 3a:", (R.subs(b=3 * a)).simplify_full())
