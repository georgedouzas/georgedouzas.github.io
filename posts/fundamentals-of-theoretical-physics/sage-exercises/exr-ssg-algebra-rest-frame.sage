# Exercise exr-ssg-algebra-rest-frame: the right side 2 sigma^mu P_mu of {Q_a, Q^dag_bdot} for a massive particle at rest,
# P^mu = (M, 0, 0, 0), and a massless one, P^mu = (E, 0, 0, E), in the metric (+, -, -, -) (P_i = -P^i). At rest the
# matrix is 2M times the identity (two pairs of fermionic oscillators); for the massless momentum it is diag(0, 4E),
# so Q_1 anticommutes to zero with its adjoint and must vanish on a positive-norm space.
M, E = var("M E", domain="positive")
pauli = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
sig = [identity_matrix(SR, 2)] + pauli; eta = [1, -1, -1, -1]
def rhs(Pup): return 2 * sum(eta[m] * Pup[m] * sig[m] for m in range(4))      # 2 sigma^mu P_mu with P_mu = eta_{mu mu} P^mu
Rm = rhs([M, 0, 0, 0]); R0 = rhs([E, 0, 0, E])
print("massive at rest: {Q, Q^dag} =", Rm.list(), "; massless along z:", R0.list())
print("[PASS] at rest {Q_a, Q^dag_b} = 2M delta_ab:", Rm == 2 * M * identity_matrix(2))
print("[PASS] massless: {Q_1, Q_1^dag} = 0 and {Q_2, Q_2^dag} = 4E:", R0 == diagonal_matrix([0, 4 * E]))
# a general momentum: the eigenvalues of 2 p.sigma are 2(E -+ |p|), both >= 0 for a physical momentum
px, py, pz = var("px py pz", domain="real"); Ep = sqrt(M^2 + px^2 + py^2 + pz^2)
Rg = rhs([Ep, px, py, pz]); ev = Rg.eigenvalues()
print("eigenvalues for a general momentum:", ev)
print("[PASS] det = 4 M^2 and trace = 4E, so both eigenvalues are positive for M > 0:",
      bool((Rg.det() - 4 * M^2).simplify_full() == 0) and bool((Rg.trace() - 4 * Ep).simplify_full() == 0))
