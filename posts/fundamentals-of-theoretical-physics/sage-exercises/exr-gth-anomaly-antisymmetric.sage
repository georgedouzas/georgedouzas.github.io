# Exercise exr-gth-anomaly-antisymmetric: the anomaly of the k-index antisymmetric representation [N, k] = Lambda^k(C^N) of
# SU(N), relative to the fundamental. For a traceless diagonal generator h = diag(a_1, ..., a_N) the weights of [N, k]
# are the sums a_{i1} + ... + a_{ik} over k-subsets, and Tr_{[N,k]}(h^3) = A_N(k) Tr_N(h^3) with Nath's eq. (6.123)
#   A_N(k) = (N - 3)! (N - 2k) / ((N - k - 1)! (k - 1)!).
# Since the cubic invariant of SU(N) is unique, one generator suffices.
from itertools import combinations
def A_formula(N, k): return factorial(N - 3) * (N - 2 * k) / (factorial(N - k - 1) * factorial(k - 1))
ok = True
for N in range(4, 10):
    R = PolynomialRing(QQ, ["a%d" % j for j in range(N - 1)]); a = list(R.gens()); a.append(-sum(a))    # traceless
    p3 = sum(x^3 for x in a)
    row = []
    for k in range(1, N):
        tr = sum(sum(c)^3 for c in combinations(a, k))
        ratio = tr / p3
        row.append(ratio)
        ok &= (ratio == A_formula(N, k))
    print("SU(%d): A_N(k) for k = 1..%d:" % (N, N - 1), [ZZ(r) for r in row])
print("[PASS] Tr_{[N,k]}(h^3) = A_N(k) Tr_N(h^3) with Nath's formula, N = 4..9:", ok)
print("[PASS] SU(5): A(5bar) + A(10) = -A_5(1) + A_5(2) = 0, so one generation 5bar + 10 is anomaly-free:", -A_formula(5, 1) + A_formula(5, 2) == 0)
print("[PASS] SU(6): 2 [6,1] + [6,4] is anomaly-free, since A_6(4) = -2:", 2 * A_formula(6, 1) + A_formula(6, 4) == 0)
print("[PASS] [N, k] and [N, N - k] are conjugate and have opposite anomalies, for all N <= 9:",
      all(A_formula(N, k) == -A_formula(N, N - k) for N in range(4, 10) for k in range(1, N)))
