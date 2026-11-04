# Exercise exr-rv-st-petersburg: the waiting time K for the first head has P(K = k) = 2^-k; E K = 2, E K^2 = 6, the
# payoff 2^K has infinite expectation, and capped at 2^30 it has expectation 31.
k = var("k")
print("E K =", sum(k * 2 ** (-k), k, 1, oo))
print("E K^2 =", sum(k ** 2 * 2 ** (-k), k, 1, oo))
print("E 2^K: the terms 2^k 2^-k are all", (2 ** k * 2 ** (-k)).simplify_full(), "so the series diverges")
cap = 30
capped = sum(QQ(2) ** j * QQ(2) ** (-j) for j in range(1, cap + 1)) + QQ(2) ** cap * sum(QQ(2) ** (-j) for j in range(cap + 1, 200))
print("expected payoff capped at 2^30 (tail truncated at k = 200):", numerical_approx(capped, digits=10), "-> 31")
