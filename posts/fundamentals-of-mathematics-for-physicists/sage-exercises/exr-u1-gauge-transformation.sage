# Exercise exr-u1-gauge-transformation: a U(1) gauge transformation A -> A + d lambda leaves the curvature F = dA
# unchanged, checked with SageManifolds on the plane for a generic potential and a generic gauge function; and the
# non-abelian curvature of a constant su(2) potential is the commutator of its components.
M = Manifold(2, "R^2"); X = M.chart("x y"); x, y = X[:]
A1, A2, lam = [function(n)(x, y) for n in ["A1", "A2", "lam"]]
A = M.one_form(A1, A2, name="A")
L = M.scalar_field(lam, name="lambda")
Ag = A + L.differential()
print("F = dA =", A.exterior_derivative().display())
print("F is gauge invariant, d(A + d lambda) = dA:", Ag.exterior_derivative() == A.exterior_derivative())
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
c1, c2 = var("c1 c2", domain="real")
Ax, Ay = c1 * T[0], c2 * T[1]                               # constant su(2) potential: A_x = c1 T_1, A_y = c2 T_2
F = Ax * Ay - Ay * Ax                                       # F_xy = d_x A_y - d_y A_x + [A_x, A_y] with constant A
print("F_xy for the constant potential = [A_x, A_y] = c1 c2 [T_1, T_2] = c1 c2 T_3:", bool(F == c1 * c2 * T[2]))
