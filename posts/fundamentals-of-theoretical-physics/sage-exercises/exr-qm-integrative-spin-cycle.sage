# Exercise exr-qm-integrative-spin-cycle: prepare |+x>, evolve under (hbar omega/2) sigma_3, measure S_x; Zeno limit.
t, w, T = var("t w T", domain="positive"); N = var("N")
sig = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
plus = vector(SR, [1, 1]) / sqrt(2)
U = diagonal_matrix(SR, [exp(-I * w * t / 2), exp(I * w * t / 2)])
psi = U * plus
bl = [(psi.conjugate() * S * psi).exponentialize().simplify_full().trig_reduce() for S in sig]
print("(a) Bloch vector:", [b.simplify_trig() for b in bl])
P = (abs(plus.conjugate() * psi) ** 2).exponentialize().simplify_full()
print("(b) Prob(S_x = +hbar/2) - cos(w t/2)^2 =", (P - cos(w * t / 2) ** 2).exponentialize().simplify_full())
lhs = (1 - bl[0] ** 2) * (1 - bl[1] ** 2) - bl[2] ** 2
print("(c) (Delta s1)^2 (Delta s2)^2 - <s3>^2 =", lhs.trig_reduce(), "(>= 0)")
print("(d) N = 1, 10, 100, 1000 with w T = pi:", [(cos(pi / (2 * n)) ** (2 * n)).n(digits=5) for n in [1, 10, 100, 1000]], "; limit:", limit(cos(w * T / (2 * N)) ** (2 * N), N=oo))
