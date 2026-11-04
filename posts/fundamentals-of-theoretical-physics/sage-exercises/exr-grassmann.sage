# Exercise exr-grassmann: Grassmann (anticommuting) variables in Sage's exterior algebra. Two generators theta_1,
# theta_2 and a spinor parameter: theta^2 = 0, theta_1 theta_2 = -theta_2 theta_1, the "spinor product" xi chi = xi^a chi_a
# with raised index xi^1 = xi_2, xi^2 = -xi_1 is symmetric, xi chi = chi xi (Martin eq. 2.16), and xi xi = 2 xi_2 xi_1 != 0.
E = ExteriorAlgebra(QQ, ["x1", "x2", "c1", "c2"]); x1, x2, c1, c2 = E.gens()
print("[PASS] theta^2 = 0 and theta_1 theta_2 = -theta_2 theta_1:", (x1 * x1).is_zero() and x1 * x2 == -(x2 * x1))
xi = [x1, x2]; chi = [c1, c2]
def up(v): return [v[1], -v[0]]
def dot(a, b): return sum(up(a)[i] * b[i] for i in range(2))
print("xi chi =", dot(xi, chi), " chi xi =", dot(chi, xi))
print("[PASS] xi chi = chi xi, no sign from reordering because of the antisymmetric epsilon (Martin eq. 2.16):", dot(xi, chi) == dot(chi, xi))
print("[PASS] xi xi = 2 xi_2 xi_1, non-zero although each component squares to zero:", dot(xi, xi) == 2 * x2 * x1)
# a function of Grassmann variables is a polynomial of degree at most the number of generators
f = 1 + 3 * x1 + x1 * x2 * c1
print("f = 1 + 3 theta_1 + theta_1 theta_2 chi_1 has degrees", sorted(set(len(k) for k in f.monomial_coefficients())), "and f * x1 * x2 * c1 * c2 =", f * x1 * x2 * c1 * c2)
print("[PASS] the top form theta_1 theta_2 chi_1 chi_2 kills every monomial of positive degree:", f * x1 * x2 * c1 * c2 == x1 * x2 * c1 * c2)
