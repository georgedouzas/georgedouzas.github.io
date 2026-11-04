# Exercise exr-ssg-fr-ads7: the other Freund-Rubin solution of eleven-dimensional supergravity, AdS_7 x S^4, with the flux on
# the four internal directions, F_{mnpq} = c eps_{mnpq} (eps of the internal metric g_mn = -gamma_mn). We work at a point in
# an orthonormal frame, eta = diag(1, -1, ..., -1), compute FF_MN and F^2 by brute force, use the trace-reversed equation
# R_MN = -(1/3)(FF_MN - (1/12) g_MN F^2), and compare with AdS_7 of radius L (R_{mu nu} = 6 g_{mu nu}/L^2 in our signature) and
# S^4 of radius a (R_mn = 3 gamma_mn/a^2). The same code reproduces AdS_4 x S^7.
from itertools import permutations
c = var("c", domain="positive")
def solve_split(dext, flux_on_internal):
    D = 11; eta = [1] + [-1] * (D - 1)
    flux_idx = list(range(dext, D)) if flux_on_internal else list(range(0, 4))
    F = {}
    for p in permutations(flux_idx):
        F[p] = c * Permutation([flux_idx.index(i) + 1 for i in p]).sign()
    def FF(M, N):
        return sum(F.get((M,) + q, 0) * F.get((N,) + q, 0) * prod(eta[i] for i in q) for q in permutations(flux_idx, int(3)))
    F2 = sum(F[p]^2 * prod(eta[i] for i in p) for p in F)
    Ric = lambda M: -(FF(M, M) - eta[M] * F2 / 12) / 3
    return [Ric(M) for M in range(D)], eta, F2
# AdS_7 x S^4: external 0..6, flux on 7..10
Ric, eta, F2 = solve_split(7, True)
k_ext = (Ric[0] / eta[0]).simplify_full(); k_int = (Ric[7] / eta[7]).simplify_full()   # R_MN = k g_MN on each factor
print("AdS_7 x S^4: F^2 =", F2, "; R_{mu nu} =", k_ext, "g_{mu nu}; R_mn =", k_int, "g_mn")
L2 = 6 / k_ext; a2 = 3 / (-k_int)                 # R_{mu nu} = 6/L^2 g (AdS_7), R_mn = -3/a^2 g_mn = 3/a^2 gamma_mn (S^4)
print("[PASS] both factors are Einstein: AdS_7 (k > 0 in our signature) and S^4 (k < 0 for g_mn = -gamma_mn):", bool(k_ext > 0) and bool(k_int < 0) and all(bool((Ric[M] / eta[M] - (k_ext if M < 7 else k_int)).simplify_full() == 0) for M in range(11)))
print("L_AdS7^2 =", L2, ", a_S4^2 =", a2)
print("[PASS] the AdS_7 radius is twice the S^4 radius:", bool((L2 / a2).simplify_full() == 4))
# AdS_4 x S^7 with the flux on spacetime: c = 3m
Ric, eta, F2 = solve_split(4, False)
k_ext = (Ric[0] / eta[0]).simplify_full(); k_int = (Ric[4] / eta[4]).simplify_full()
L2 = 3 / k_ext; a2 = 6 / (-k_int)
print("AdS_4 x S^7: R_{mu nu} =", k_ext, "g; R_mn =", k_int, "g; L^2 =", L2, ", a^2 =", a2)
print("[PASS] with c = 3m this is R_{mu nu} = 12 m^2 g, R_mn = -6 m^2 g, and the S^7 radius is twice the AdS_4 radius:",
      bool(k_ext.subs(c=3 * var("m")) == 12 * var("m")^2) and bool(k_int.subs(c=3 * var("m")) == -6 * var("m")^2) and bool((a2 / L2).simplify_full() == 4))
