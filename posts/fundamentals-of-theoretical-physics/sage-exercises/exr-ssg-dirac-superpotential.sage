# Exercise exr-ssg-dirac-superpotential: two chiral multiplets with W = m Phi_1 Phi_2. The F-terms are F_1 = -m^* phi_2^*,
# F_2 = -m^* phi_1^*, the potential |m|^2(|phi_1|^2 + |phi_2|^2), and the fermion mass matrix W^{ij} = [[0, m], [m, 0]],
# whose squared singular values are |m|^2 twice: the two Weyl fermions form one Dirac fermion of mass |m|, degenerate
# with the two complex scalars.
m = var("m"); mb = var("mb")                                   # m and its conjugate as independent symbols
p1, p2, c1, c2 = var("p1 p2 c1 c2")                            # phi_i and phi_i^*
W = m * p1 * p2; Wb = mb * c1 * c2
Wi = [diff(W, p) for p in (p1, p2)]; Wbi = [diff(Wb, c) for c in (c1, c2)]
print("W^i =", Wi, "; F_i = -conj(W^i) =", [-w for w in Wbi])
V = sum(Wi[i] * Wbi[i] for i in range(2))
print("[PASS] V = |m|^2 (|phi_1|^2 + |phi_2|^2):", bool((V - m * mb * (p1 * c1 + p2 * c2)).expand() == 0))
M = matrix(SR, 2, 2, lambda i, j: diff(W, (p1, p2)[i], (p1, p2)[j]))
print("fermion mass matrix W^{ij} =", M.list())
mr = var("mr", domain="positive")
Mr = M.subs(m=mr)
print("[PASS] M^dag M = |m|^2 * identity: both Weyl fermions have mass |m|, the scalar mass too:", Mr.transpose() * Mr == mr^2 * identity_matrix(2))
print("[PASS] the scalar mass matrix d^2V/dphi^*_i dphi_j = |m|^2 * identity equals M^dag M:",
      matrix(SR, 2, 2, lambda i, j: diff(V, (c1, c2)[i], (p1, p2)[j])).subs(m=mr, mb=mr) == Mr.transpose() * Mr)
