# Exercise exr-ssg-killing-s2: Killing spinors on the unit two-sphere, D_m eta = (i/2) s gamma_m eta with s = +1 or -1.
# Metric d theta^2 + sin^2 theta d phi^2, zweibein e^1 = d theta, e^2 = sin theta d phi, spin connection
# omega^{12} = -cos theta d phi, D_m = d_m + (1/4) omega_m^{ab} gamma_ab with gamma_1 = sigma_1, gamma_2 = sigma_2.
# For each sign the solutions form a two-dimensional space (the maximal number for a 2-component spinor): S^2 is maximally
# supersymmetric in this sense, as the round S^7 is for the analogous equation, with 8 = 2^3 solutions (Duff et al.).
th, ph = var("theta phi", domain="real")
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
g12 = s1 * s2                                   # gamma_{12} = gamma_1 gamma_2 = i sigma_3
def Dth(v): return diff(v, th)
def Dph(v): return diff(v, ph) + (-cos(th)) * (g12 / 2) * v          # (1/4) omega^{ab} gamma_ab = (1/2) omega^{12} gamma_12
gam_th = s1; gam_ph = sin(th) * s2              # gamma_m = e_m^a gamma_a
results = {}
for sgn in (1, -1):
    sols = []
    for eta0 in (vector(SR, [1, 0]), vector(SR, [0, 1])):
        U = cos(th / 2) * identity_matrix(SR, 2) + I * sgn * sin(th / 2) * s1            # exp(i s theta sigma_1 / 2)
        Vp = cos(ph / 2) * identity_matrix(SR, 2) + sin(ph / 2) * g12                     # exp(phi gamma_12 / 2), gamma_12^2 = -1
        eta = U * Vp * eta0
        ta, tb = var("ta tb")                                                             # half angles: theta = 2 ta, phi = 2 tb
        zero = lambda w: all(bool(c.subs({th: 2 * ta, ph: 2 * tb}).expand_trig().simplify_full() == 0) for c in w)
        ok = zero(Dth(eta) - I * sgn / 2 * gam_th * eta) and zero(Dph(eta) - I * sgn / 2 * gam_ph * eta)
        sols.append((eta, ok))
    results[sgn] = sols
    print("s = %+d:" % sgn, [ok for _, ok in sols])
print("[PASS] for both signs, eta = exp(i s theta sigma_1 / 2) exp(phi gamma_12 / 2) eta_0 solves D_m eta = (i s/2) gamma_m eta for every constant eta_0:",
      all(ok for sgn in results for _, ok in results[sgn]))
M = matrix(SR, [list(results[1][0][0]), list(results[1][1][0])])
print("[PASS] the two solutions for s = +1 are linearly independent (2 Killing spinors, the maximum):", bool(M.det().simplify_full() != 0))
