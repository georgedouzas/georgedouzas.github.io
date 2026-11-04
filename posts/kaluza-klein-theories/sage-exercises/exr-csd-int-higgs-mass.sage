# Exercise exr-csd-int-higgs-mass: the tree-level spectrum of the reduced theory with
# L = (1/g^2) [ (1/2) Tr F_mu nu F^mu nu - (1/R^2) sum_a Tr(D phi_a D phi_a) - V/R^4 ],  V = -Tr F_12^2.
# Scalar: q = 1 + rho with kinetic term (1/(g^2 R^2)) (d rho)^2 and potential (1/(g^2 R^4)) V(1 + rho).
# Vectors: after A -> g A the mass term is (1/2) A^i A^j M^2_ij / R^2 with M^2_ij = -2 sum_a Tr([T_i, phi_a][T_j, phi_a]).
g, R, rho = var("g R rho", domain="positive")
Vr = lambda r: r^2 / 2 - r + 2/3
Vq = taylor(Vr((1 + rho)^2), rho, 0, 2)
A_kin = 1 / (g^2 * R^2); B_mass = Vq.coefficient(rho, 2) / (g^2 * R^4)    # L = A (d rho)^2 - B rho^2 + ...
mH2 = (B_mass / A_kin).simplify_full()
print("V(1 + rho) =", Vq, "+ O(rho^3);  m_H^2 = B/A =", mH2)
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
ad = lambda X, Y: X * Y - Y * X
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
def up(m): M = zero_matrix(SR, 3, 3); M[0:2, 0:2] = m; return M
T = [I / 2 * up(m) for m in s] + [I / (2 * sqrt(3)) * diagonal_matrix(SR, [1, 1, -2])]
Pp = E(1, 2); Pm = -E(2, 1); f = [(Pp + Pm) / 2, I * (Pp - Pm) / 2]
M2 = matrix(SR, 4, 4, lambda i, j: -2 * sum((ad(T[i], x) * ad(T[j], x)).trace() for x in f)) / R^2
ev = sorted(M2.eigenvalues(), key=lambda z: (z * R^2))
MW2, MZ2 = ev[1], ev[3]
print("gauge boson masses^2:", ev)
print("[PASS] m_H^2 = 2/R^2, M_W^2 = 1/(2 R^2), M_Z^2 = 2/R^2: m_H = 2 M_W = M_Z, independent of g and R:",
      bool(mH2 == 2 / R^2) and bool(MW2 == 1 / (2 * R^2)) and bool(MZ2 == 2 / R^2) and bool((mH2 / MW2).simplify_full() == 4))
# the same result in Standard Model language: v^2 = 2/(g^2 R^2), lambda = g^2/2 in V = lambda (|H|^2 - v^2/2)^2
v2 = 2 / (g^2 * R^2); lam = g^2 / 2
print("[PASS] with H = Phi/(g R): v^2 = 2/(g^2 R^2), lambda = g^2/2, m_H^2 = 2 lambda v^2 and M_W^2 = g^2 v^2/4 reproduce the values above:",
      bool((2 * lam * v2 - mH2).simplify_full() == 0) and bool((g^2 * v2 / 4 - MW2).simplify_full() == 0))
