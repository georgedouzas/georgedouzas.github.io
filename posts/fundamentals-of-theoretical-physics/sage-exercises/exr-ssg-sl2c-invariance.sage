# Exercise exr-ssg-sl2c-invariance: a Lorentz transformation acts on left-handed spinors by a matrix M of SL(2, C),
# xi -> M xi. The product xi chi = xi^a chi_a = -xi^T eps chi (lower indices, eps^{12} = 1) is invariant because
# M^T eps M = det(M) eps. The boost along z with rapidity b is M = exp(-b sigma_3/2) on left-handed spinors, and
# M sigma^mu M^dag = Lambda^mu_nu sigma^nu reproduces the vector boost.
a, b, c, d = var("a b c d")
M = matrix(SR, [[a, b], [c, d]]); eps = matrix(SR, [[0, 1], [-1, 0]])
print("[PASS] M^T eps M = det(M) eps for a general 2x2 matrix, so the product is invariant iff det M = 1:",
      bool(((M.transpose() * eps * M) - M.det() * eps).expand() == 0))
r = var("r", domain="real")
pauli = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
sig = [identity_matrix(SR, 2)] + pauli
Mb = matrix(SR, [[exp(-r / 2), 0], [0, exp(r / 2)]])            # exp(-r sigma_3 / 2)
print("[PASS] the boost matrix has unit determinant:", bool(Mb.det() == 1))
# x^mu sigma_mu with sigma_mu = (1, -sigma_i): X = x^0 - x.sigma; X -> M X M^dag
x0, x1, x2, x3 = var("x0 x1 x2 x3", domain="real")
X = x0 * sig[0] - x1 * pauli[0] - x2 * pauli[1] - x3 * pauli[2]
Xp = (Mb * X * Mb.conjugate_transpose()).simplify_full()
print("det X =", X.det().expand(), "(the Minkowski norm)")
print("[PASS] det X = x0^2 - x1^2 - x2^2 - x3^2 is preserved:", bool((Xp.det() - X.det()).simplify_full() == 0))
t_new = ((Xp[0, 0] + Xp[1, 1]) / 2).simplify_full(); z_new = (-(Xp[0, 0] - Xp[1, 1]) / 2).simplify_full()
print("t' =", t_new, " z' =", z_new)
ch, sh = (exp(r) + exp(-r)) / 2, (exp(r) - exp(-r)) / 2
print("[PASS] the transformation is the boost t' = cosh(r) t + sinh(r) z, z' = sinh(r) t + cosh(r) z:",
      bool((t_new - ch * x0 - sh * x3).simplify_full() == 0) and bool((z_new - sh * x0 - ch * x3).simplify_full() == 0))
