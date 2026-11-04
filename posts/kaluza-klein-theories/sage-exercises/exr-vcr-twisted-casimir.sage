# Exercise exr-vcr-twisted-casimir: the regularised sum over the twisted tower, sum_{n in Z} |n + alpha| for 0 < alpha < 1.
# With an exponential cutoff the sum is 2/eps^2 + C(alpha) + O(eps); compute the finite part C(alpha) and compare with
# the Hurwitz zeta values zeta(-1, alpha) + zeta(-1, 1 - alpha) = -B_2(alpha).
eps, al = var("eps alpha")
S1 = -diff(exp(-eps * al) / (1 - exp(-eps)), eps)              # sum_{n >= 0} (n + alpha) e^{-eps (n + alpha)}
S2 = -diff(exp(-eps * (1 - al)) / (1 - exp(-eps)), eps)        # sum_{n >= 1} (n - alpha) e^{-eps (n - alpha)}
ser = (S1 + S2).series(eps, 2).truncate().expand()
print("cutoff expansion:", ser.collect(eps))
C = ser.coefficient(eps, 0)
print("finite part C(alpha) =", C.factor(), "; equals -(alpha^2 - alpha + 1/6):", bool((C + (al ^ 2 - al + 1 / 6)).expand() == 0))
import mpmath
for a in [1 / 4, 1 / 3, 1 / 2]:
    print("alpha = %s: zeta(-1, a) + zeta(-1, 1 - a) = %.10f, -B_2(alpha) = %.10f" % (a, float(mpmath.zeta(-1, a) + mpmath.zeta(-1, 1 - a)), float(-bernoulli_polynomial(QQ(a), 2))))
print("alpha -> 0: C -> -1/6 = 2 zeta(-1), the untwisted value")
