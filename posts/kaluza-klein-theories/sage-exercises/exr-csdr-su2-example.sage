# Exercise exr-csdr-su2-example: Kubyshin's example, SU(2) Yang-Mills on S^2 = SU(2)/U(1) with the embedding
# tau_n(Q_3) = n Q_3 (hep-th/0304...; notes: eqs. 30-37). For n = 1 the constraint leaves one complex parameter f,
# the curvature is F = -i sigma_3 (|f|^2 - 1) sin th dth ^ dph and the action is proportional to (|f|^2 - 1)^2.
Q = [matrix(SR, [[0, 1], [1, 0]]) / (2 * I), matrix(SR, [[0, -I], [I, 0]]) / (2 * I), matrix(SR, [[1, 0], [0, -1]]) / (2 * I)]
ad = lambda A, B: A * B - B * A
X = matrix(SR, 2, 2, lambda a, b: var("x%d%d" % (a, b)))
for n in (1, 2):
    sol = solve([e == 0 for e in (ad(n * Q[2], X) - I * X).list()], X.list(), solution_dict=True)[0]
    print("n = %d: +i eigenspace of ad(n Q_3) has free entries" % n, [v for v in X.list() if sol[v] != 0])
free1 = [v for v in X.list() if solve([e == 0 for e in (ad(Q[2], X) - I * X).list()], X.list(), solution_dict=True)[0][v] != 0]
free2 = [v for v in X.list() if solve([e == 0 for e in (ad(2 * Q[2], X) - I * X).list()], X.list(), solution_dict=True)[0][v] != 0]
print("[PASS] for n = 1 the +i eigenspace is one-dimensional (E_21 in these conventions: one complex parameter f), for n = 2 it is empty (phi = 0, a pure monopole):",
      free1 == [X[1, 0]] and free2 == [])
f, fb = var("f fb")
E12 = matrix(SR, [[0, 1], [0, 0]]); E21 = matrix(SR, [[0, 0], [1, 0]])
phip = f * E21; phim = -fb * E12                     # phi_+ in the +i eigenspace, phi_- = -phi_+^dagger
phi1 = (phip + phim) / 2; phi2 = I * (phip - phim) / 2
F12 = (ad(phi1, phi2) - Q[2]).simplify_full()
print("F_12 =", F12)
sig3 = diagonal_matrix(SR, [1, -1])
print("[PASS] F_12 is proportional to (|f|^2 - 1) sigma_3, so the potential -Tr F_12^2 = (|f|^2 - 1)^2/2 has its minima at |f| = 1 (Kubyshin eqs. 36-37):",
      bool((F12 - F12[0, 0] * sig3).is_zero()) and bool((F12[0, 0] / (f * fb - 1)).simplify_full().is_constant()) and bool((-(F12 * F12).trace() - (f * fb - 1)^2 / 2).expand() == 0))
