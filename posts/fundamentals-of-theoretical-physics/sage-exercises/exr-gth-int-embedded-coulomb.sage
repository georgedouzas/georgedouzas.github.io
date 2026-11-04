# Exercise exr-gth-int-embedded-coulomb: an SU(2) potential pointing along a fixed generator, A_mu = a_mu T_3 with the
# Coulomb potential a_0 = Q/r, a_i = 0, has F = (da) T_3; the Yang-Mills equations D_mu F^{mu nu} = 0 reduce to
# Maxwell's equations and hold away from r = 0; the doublet components couple with charges -1/2 and +1/2.
t, x, y, z, Q = var("t x y z Q", domain="real"); X = [t, x, y, z]
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
r = sqrt(x^2 + y^2 + z^2)
a = [Q / r, 0, 0, 0]
A = [a[m] * T[2] for m in range(4)]
eta = [1, -1, -1, -1]
F = [[diff(A[n], X[m]) - diff(A[m], X[n]) + A[m] * A[n] - A[n] * A[m] for n in range(4)] for m in range(4)]
print("[PASS] the commutator terms vanish: F = (d_mu a_nu - d_nu a_mu) T_3:", all((F[m][n] - (diff(a[n], X[m]) - diff(a[m], X[n])) * T[2]).is_zero() for m in range(4) for n in range(4)))
Fup = [[eta[m] * eta[n] * F[m][n] for n in range(4)] for m in range(4)]
YM = [sum((diff(Fup[m][n], X[m]) + A[m] * Fup[m][n] - Fup[m][n] * A[m] for m in range(4)), zero_matrix(SR, 2)).apply_map(lambda e: e.simplify_full()) for n in range(4)]
print("[PASS] D_mu F^{mu nu} = 0 for r != 0 (Laplace's equation for Q/r):", all(Y.is_zero() for Y in YM))
print("the electric field E^i = F^{i0} along T_3:", [(-2 * (T[2] * Fup[i][0]).trace()).simplify_full() for i in range(1, 4)])
print("D_mu on a doublet: d_mu + a_mu T_3 = d_mu + i q a_mu with q =", [(-I / 2 * s[2])[k, k] / I for k in range(2)])
