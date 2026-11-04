# Exercise exr-freund-rubin: the Freund-Rubin ansatz F_{mu nu rho sigma} = 3m epsilon_{mu nu rho sigma} on M_4 x X_7.
# With the eleven-dimensional field equations R_MN - g_MN R/2 = -(1/3)(F_MPQR F_N^PQR - g_MN F^2/8) of the Lagrangian
# -e R/4 - e F^2/48 in our signature (+, -, ..., -) (the right side has the opposite sign in the mostly-plus metric of
# Duff-Nilsson-Pope, arXiv:2502.07710, eq. 7), compute the contractions with the Levi-Civita symbol in Lorentzian
# signature and deduce R_{mu nu} = 12 m^2 g_{mu nu} (anti-de Sitter) and R_{mn} = -6 m^2 g_{mn} (a sphere, whose
# metric g_mn = -gamma_mn is negative definite).
m = var("m")
eta = diagonal_matrix([1, -1, -1, -1])
def levi(*idx): return Permutation([i + 1 for i in idx]).sign() if len(set(idx)) == 4 else 0
R4 = range(4)
# F_{mu PQR} F_nu^{PQR}: three indices raised with eta (diagonal)
FF = matrix(SR, 4, 4, lambda mu, nu: (3 * m)^2 * sum(levi(mu, a, b, c) * levi(nu, a, b, c) * eta[a, a] * eta[b, b] * eta[c, c] for a in R4 for b in R4 for c in R4))
F2 = (3 * m)^2 * sum(levi(a, b, c, d)^2 * eta[a, a] * eta[b, b] * eta[c, c] * eta[d, d] for a in R4 for b in R4 for c in R4 for d in R4)
print("F_{mu PQR} F_nu^{PQR} =", FF.diagonal(), " F^2 =", F2)
print("[PASS] epsilon_{mu abc} epsilon_nu^{abc} = -6 eta_{mu nu} and epsilon^2 = -24 in Lorentzian signature (they depend only on det eta = -1):", bool(FF == -54 * m^2 * eta) and bool(F2 == -216 * m^2))
# eleven-dimensional trace: R - 11 R/2 = T, T = -(1/3)(F^2 - 11 F^2/8)
T = -(F2 - 11 * F2 / 8) / 3; R11 = -2 * T / 9
Rmunu = -(FF - eta * F2 / 8) / 3 + eta * R11 / 2
g7 = var("g7")                                                        # a generic internal metric component g_mn
Rmn = -(0 - g7 * F2 / 8) / 3 + g7 * R11 / 2
print("R =", R11, "; R_{mu nu} =", Rmunu.diagonal(), "; R_mn =", Rmn.expand())
print("[PASS] R_{mu nu} = 12 m^2 g_{mu nu} (AdS_4) and R_mn = -6 m^2 g_mn (S^7 with g_mn = -gamma_mn), DNP eqs. 10-11 in our signature:", Rmunu == 12 * m^2 * eta and bool(Rmn.expand() == -6 * m^2 * g7))
# the seven-sphere of radius a has R_mn = 6 gamma_mn / a^2, so a = 1/m; AdS_4 has R_{mu nu} = 3 g_{mu nu} / L^2 in our signature, so L = 1/(2m)
print("[PASS] the radii: sphere a = 1/m and anti-de Sitter L = 1/(2m), the sphere twice as large as the AdS scale:", bool(6 / (1 / m)^2 == 6 * m^2) and bool(3 / (1 / (2 * m))^2 == 12 * m^2))
