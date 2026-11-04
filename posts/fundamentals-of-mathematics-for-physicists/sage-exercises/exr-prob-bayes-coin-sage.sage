# Exercise exr-prob-bayes-coin-sage: posterior probability that a coin drawn from a bag of 999 fair coins and one
# two-headed coin is the two-headed one, after k heads in a row; exact rationals, and the first k with posterior > 0.99.
prior = QQ(1) / 1000
post = lambda k: prior / (prior + (1 - prior) * QQ(2) ** (-k))
for k in [0, 1, 5, 8, 10, 12, 15, 17, 20]:
    print("k =", k, ": posterior =", post(k), "=", numerical_approx(post(k), digits=4))
first = next(k for k in range(0, 40) if post(k) > QQ(99) / 100)
print("smallest k with posterior above 0.99:", first, "; log_2(999 * 99) =", numerical_approx(log(999 * 99, 2), digits=4))
print("posterior odds at k = 17: 2^17/999 =", QQ(2) ** 17 / 999, "> 99:", QQ(2) ** 17 / 999 > 99)
