# Exercise exr-ssg-weyl-plane-wave: plane-wave solutions psi = u exp(-i p.x) of the massless Weyl equation
# i sigmabar^mu d_mu psi = 0, i.e. (p_mu sigmabar^mu) u = 0. A non-zero solution needs det(p.sigmabar) = p^2 = 0; for
# p^mu = (E, 0, 0, E) the solution space is one-dimensional, spanned by u = (0, 1): one helicity state, which is why a
# Weyl spinor has 2 real on-shell degrees of freedom (the complex amplitude) against 4 off-shell.
E = var("E", domain="positive"); p0, p3 = var("p0 p3")
pauli = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
sigb = [identity_matrix(SR, 2)] + [-q for q in pauli]; eta = [1, -1, -1, -1]
def pdot(pup): return sum(eta[m] * pup[m] * sigb[m] for m in range(4))    # p_mu sigmabar^mu with p_mu = eta_{mu mu} p^mu
Mg = pdot([p0, 0, 0, p3])
print("det(p.sigmabar) =", Mg.det().expand())
print("[PASS] a non-zero plane wave needs det = p0^2 - p3^2 = 0, the massless shell:", bool(Mg.det().expand() == p0^2 - p3^2))
Mn = pdot([E, 0, 0, E]); ker = Mn.right_kernel()
print("p.sigmabar for p = (E, 0, 0, E):", Mn.list(), "; kernel:", ker.basis())
print("[PASS] the solution space is one-dimensional, spanned by u = (0, 1):", ker.dimension() == 1 and ker.basis()[0] == vector(SR, [0, 1]))
# helicity: the rotation generator about the momentum on left-handed spinors is sigma_3/2; u = (0, 1) has eigenvalue -1/2
print("[PASS] u is an eigenvector of S^{12} = sigma_3/2 with eigenvalue -1/2: the left-handed spinor has helicity -1/2:",
      pauli[2] / 2 * vector(SR, [0, 1]) == -vector(SR, [0, 1]) / 2)
