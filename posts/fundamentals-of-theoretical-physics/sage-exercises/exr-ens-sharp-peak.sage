# Exercise exr-ens-sharp-peak: two Einstein solids with N_A = 300 and N_B = 200 oscillators share q = 100 quanta.
# The probability of q_A quanta in A is Omega_A(q_A) Omega_B(q - q_A)/Omega_total; find its maximum, mean and width.
NA, NB, q = 300, 200, 100
Om = lambda N, k: binomial(k + N - 1, k)
w = [Om(NA, k) * Om(NB, q - k) for k in range(q + 1)]
tot = sum(w)
print("Omega_total =", tot, "= binomial(q + N_A + N_B - 1, q):", tot == binomial(q + NA + NB - 1, q))
P = [wk / tot for wk in w]
kmax = max(range(q + 1), key=lambda k: P[k])
mean = sum(k * P[k] for k in range(q + 1)); var_ = sum(k ** 2 * P[k] for k in range(q + 1)) - mean ** 2
print("most probable q_A =", kmax, "; mean =", mean.n(digits=6), "(q N_A/(N_A + N_B) = %s)" % (q * NA / (NA + NB)), "; standard deviation =", sqrt(var_).n(digits=4))
print("P(q_A in [50, 70]) =", sum(P[50:71]).n(digits=6), "; P(q_A <= 30) =", sum(P[:31]).n(digits=3))
