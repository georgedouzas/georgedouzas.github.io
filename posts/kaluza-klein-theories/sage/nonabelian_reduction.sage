# Oracle (task 5.3): the curvature scalar of the non-abelian Kaluza-Klein ansatz on M4 x S^2 computed with
# SageManifolds and compared with R_6 = R_4 + R_h - (1/4) h_{mn} K^m_a K^n_b F^a_{mu nu} F^{b mu nu}, where the internal
# block of the metric is h = -gamma (signature (+,-,-,-,-,-), gamma the round metric), so R_h = -R_gamma = -2 and
# R_6 = R_4 - 2 + (1/4) gamma_{mn} K^m_a K^n_b F^a F^b, with
# F^c = dA^c + eps_{abc} A^a A^b: the gauge algebra has structure constants f_ab^c = +eps_abc, opposite to the Killing
# brackets [K_a, K_b] = -eps_abc K_c of Part I (the Killing vectors act from the right), as the gauge transformation
# delta A^c = d eps^c + eps_abc A^a eps^b of Gauge.lean also shows. Both signs are tested; only + passes. Spacetime is flat and the gauge fields are explicit polynomials in (t, x) with three
# non-zero components so that the commutator term is present. The identity holds pointwise on the sphere, which
# is why the ansatz is not a solution of the six-dimensional vacuum equations (the consistency problem).
M = Manifold(6, "M", structure="Lorentzian"); X = M.chart("t x y z th ph"); t, x, y, z, th, ph = X[:]
set_random_seed(5)
def rpoly(): return sum(QQ.random_element(num_bound=3, den_bound=2) * mnm for mnm in (1, t, x, t * x))
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]   # Part I conventions
gam = diagonal_matrix(SR, [1, sin(th)^2]); hK = -gam                                                  # h = -gamma
A = [[0, rpoly(), 0, 0], [rpoly(), 0, 0, 0], [0, rpoly(), 0, 0]]        # A^a_mu: A^1_x, A^2_t, A^3_x
eta = [1, -1, -1, -1]
B = [[sum(K[a][m] * A[a][mu] for a in range(3)) for mu in range(4)] for m in range(2)]     # B^m_mu = K^m_a A^a_mu
g = M.metric("g")
for mu in range(4):
    for nu in range(mu, 4): g[mu, nu] = (eta[mu] if mu == nu else 0) + sum(hK[m, n] * B[m][mu] * B[n][nu] for m in range(2) for n in range(2))
    for m in range(2): g[mu, 4 + m] = sum(hK[m, n] * B[n][mu] for n in range(2))
for m in range(2):
    for n in range(m, 2): g[4 + m, 4 + n] = hK[m, n]
print("computing the six-dimensional curvature scalar ...")
R6 = g.ricci_scalar().expr()
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
cv = [t, x, y, z]
def F(c, s):
    return [[diff(A[c][nu], cv[mu]) - diff(A[c][mu], cv[nu]) + s * sum(lc(a, b, c) * A[a][mu] * A[b][nu] for a in range(3) for b in range(3)) for nu in range(4)] for mu in range(4)]
KK = matrix(SR, 3, 3, lambda a, b: (K[a] * gam * K[b]).simplify_full())                               # K_a . K_b with gamma
for s in (-1, 1):
    Fs = [F(c, s) for c in range(3)]
    F2 = sum(KK[a, b] * eta[mu] * eta[nu] * Fs[a][mu][nu] * Fs[b][mu][nu] for a in range(3) for b in range(3) for mu in range(4) for nu in range(4))
    target = 0 - 2 + F2 / 4
    print("[%s] R_6 = R_4 - 2 + (1/4) K_a.K_b F^a F^b with F^c = dA^c %+d eps_abc A^a A^b:" % ("PASS" if s == 1 else "info", s), bool((R6 - target).simplify_full() == 0))
print("[PASS] R_6 depends on the point of the sphere (the ansatz is not a vacuum solution): R_6 at the poles differs from the equator:",
      bool((R6.subs(th=pi/2, ph=0) - R6.subs(th=pi/4, ph=0)).simplify_full() != 0))
