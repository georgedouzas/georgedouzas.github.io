# Exercise exr-nak-curvature-interference: the sign of the commutator term tested by interference. For A^3_x = c t,
# A^1_t = a, A^2_x = b (a, b constant) the derivative and commutator terms of F^3_{tx} add: F^3_{tx} = c + s a b and
# F^2_{tx} = -s a c t, with s = +1 for the field strength F^c = dA^c + eps_abc A^a A^b of the gauge transformation and
# s = -1 for the Killing-bracket sign. SageManifolds (signature (+,-,-,-,-,-)) shows that
# R_6 = -2 + (1/4) K_a . K_b F^a_{mu nu} F^{b mu nu} = -2 - (1/2) K_a . K_b F^a_{tx} F^b_{tx} holds for s = +1 only.
M = Manifold(6, "M", structure="Lorentzian"); X = M.chart("t x y z th ph"); t, x, y, z, th, ph = X[:]
a, b, c = var("a b c")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
gam = diagonal_matrix(SR, [1, sin(th)^2])
A = [[a, 0, 0, 0], [0, b, 0, 0], [0, c * t, 0, 0]]
B = [[sum(K[i][m] * A[i][mu] for i in range(3)) for mu in range(4)] for m in range(2)]
eta = [1, -1, -1, -1]
g = M.metric("g")
for mu in range(4):
    for nu in range(mu, 4): g[mu, nu] = (eta[mu] if mu == nu else 0) - sum(gam[m, n] * B[m][mu] * B[n][nu] for m in range(2) for n in range(2))
    for m in range(2): g[mu, 4 + m] = -sum(gam[m, n] * B[n][mu] for n in range(2))
for m in range(2):
    for n in range(m, 2): g[4 + m, 4 + n] = -gam[m, n]
R6 = g.ricci_scalar().expr().simplify_full()
KK = matrix(SR, 3, 3, lambda i, j: K[i] * gam * K[j])
def lc(i, j, k): return Permutation([i + 1, j + 1, k + 1]).sign() if len({i, j, k}) == 3 else 0
tt, xx = var("tt xx")
def Ftx(s):                                            # F^c_{tx} with commutator sign s
    Af = [[A[i][mu].subs(t=tt) for mu in range(2)] for i in range(3)]
    return [diff(Af[k][1], tt) - diff(Af[k][0], xx) + s * sum(lc(i, j, k) * Af[i][0] * Af[j][1] for i in range(3) for j in range(3)) for k in range(3)]
for s in [1, -1]:
    F = vector([f.subs(tt=t) for f in Ftx(s)])
    print("s = %+d: F_tx =" % s, F)
    pred = -2 - F * KK * F / 2
    print("[PASS] R_6 = -2 - (1/2) K_a . K_b F^a_tx F^b_tx with s = %+d:" % s if s == 1 else "R_6 = -2 - (1/2) K_a . K_b F^a_tx F^b_tx with s = %+d (expected False):" % s, bool((R6 - pred).simplify_full() == 0))
