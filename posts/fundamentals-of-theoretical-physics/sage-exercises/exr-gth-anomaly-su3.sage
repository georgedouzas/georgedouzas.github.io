# Exercise exr-gth-anomaly-su3: the anomaly coefficients d_abc(R) = Tr_R(t_a {t_b, t_c}) of SU(3) with the Hermitian
# generators t_a = lambda_a/2 (lambda_a the Gell-Mann matrices; t_a = i T_a for the anti-Hermitian T_a of the book).
# For the fundamental 3 we list the non-zero values with a <= b <= c; the conjugate 3bar has generators -t_a^T and
# opposite d_abc; the adjoint 8, a real representation, has d_abc = 0.
from itertools import combinations_with_replacement as cwr
L.<z> = CyclotomicField(12); i = z^3; s3 = 2 * z - z^3        # i and sqrt(3) in Q(zeta_12)
assert i^2 == -1 and s3^2 == 3
lam = [matrix(L, 3, 3, l) for l in [
    [0, 1, 0, 1, 0, 0, 0, 0, 0], [0, -i, 0, i, 0, 0, 0, 0, 0], [1, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, -i, 0, 0, 0, i, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, -i, 0, i, 0], [1 / s3, 0, 0, 0, 1 / s3, 0, 0, 0, -2 / s3]]]
t = [l / 2 for l in lam]
print("[PASS] Tr(t_a t_b) = delta_ab / 2:", all((t[a] * t[b]).trace() == (1 / 2 if a == b else 0) for a in range(8) for b in range(8)))
def d(ts, a, b, c): return (ts[a] * (ts[b] * ts[c] + ts[c] * ts[b])).trace()
nz = {(a + 1, b + 1, c + 1): d(t, a, b, c) for a, b, c in cwr(range(8), 3) if d(t, a, b, c) != 0}
for k, v in nz.items(): print("d_%d%d%d(3) = %s" % (k + (v * 4 * s3 if 8 in k else v * 4,)) + ("/(4 sqrt3)" if 8 in k else "/4"))
print("[PASS] d_118 = d_228 = d_338 = 1/(2 sqrt3) and d_888 = -1/(2 sqrt3) for the 3:",
      all(nz[(j, j, 8)] == 1 / (2 * s3) for j in (1, 2, 3)) and nz[(8, 8, 8)] == -1 / (2 * s3))
tbar = [-x.transpose() for x in t]
print("[PASS] the conjugate 3bar (generators -t^T) has d_abc(3bar) = -d_abc(3) for all a, b, c:",
      all(d(tbar, a, b, c) == -d(t, a, b, c) for a, b, c in cwr(range(8), 3)))
f = [[[-2 * i * ((t[a] * t[b] - t[b] * t[a]) * t[c]).trace() for c in range(8)] for b in range(8)] for a in range(8)]   # [t_a, t_b] = i f_abc t_c
ad = [matrix(L, 8, 8, lambda b, c: -i * f[a][b][c]) for a in range(8)]
print("[PASS] the adjoint matrices (t_a)_{bc} = -i f_abc satisfy [t_a, t_b] = i f_abc t_c:",
      all(ad[a] * ad[b] - ad[b] * ad[a] == sum(i * f[a][b][c] * ad[c] for c in range(8)) for a in range(8) for b in range(8)))
print("[PASS] the adjoint 8 is real and all its anomaly coefficients vanish:", all(d(ad, a, b, c) == 0 for a, b, c in cwr(range(8), 3)))
