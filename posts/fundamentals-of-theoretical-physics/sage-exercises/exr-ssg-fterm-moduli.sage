# Exercise exr-ssg-fterm-moduli: W = Phi_0 (Phi_1 Phi_2 - mu^2). The F-terms vanish exactly on phi_0 = 0, phi_1 phi_2 = mu^2,
# a one-complex-dimensional family of supersymmetric vacua (a modulus). At the vacuum phi_1 = phi_2 = mu the fermion
# mass matrix W^{ij} has rank 2: two massive Weyl fermions and one massless one, the superpartner of the modulus.
mu = var("mu", domain="positive"); p0, p1, p2 = var("p0 p1 p2")
W = p0 * (p1 * p2 - mu^2)
Wi = [diff(W, p) for p in (p0, p1, p2)]
print("W^i =", Wi)
sols = solve(Wi, [p0, p1, p2], solution_dict=True)
print("solutions of W^i = 0:", sols)
t = var("t")
print("[PASS] every point phi_0 = 0, phi_1 = t, phi_2 = mu^2/t (t != 0) is a supersymmetric vacuum:",
      all(bool(w.subs(p0=0, p1=t, p2=mu^2 / t).simplify_full() == 0) for w in Wi))
print("[PASS] phi_0 != 0 is never a vacuum (W^1 = phi_0 phi_2 and W^2 = phi_0 phi_1 force phi_1 = phi_2 = 0, then W^0 = -mu^2):",
      bool(Wi[0].subs(p1=0, p2=0) == -mu^2))
M = matrix(SR, 3, 3, lambda i, j: diff(W, (p0, p1, p2)[i], (p0, p1, p2)[j])).subs(p0=0, p1=mu, p2=mu)
print("W^{ij} at (0, mu, mu):", M.list(), " rank", M.rank(), " eigenvalues of M^T M:", (M.transpose() * M).eigenvalues())
print("[PASS] rank 2: masses squared 2 mu^2, 2 mu^2 and one massless multiplet, the modulus:", M.rank() == 2 and sorted((M.transpose() * M).eigenvalues(), key=str) == sorted([0, 2 * mu^2, 2 * mu^2], key=str))
# the massless direction is tangent to the vacuum curve: (0, 1, -1) at phi_1 = phi_2 = mu
print("[PASS] the null vector of W^{ij} is (0, 1, -1), the tangent to phi_1 phi_2 = mu^2 at (mu, mu):", (M * vector([0, 1, -1])).is_zero())
