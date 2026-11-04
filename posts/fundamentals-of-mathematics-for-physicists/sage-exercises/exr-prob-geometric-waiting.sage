# Exercise exr-prob-geometric-waiting: the first head appears at toss k with probability 2^-k; the probabilities sum
# to 1 (geometric series), so a head appears eventually with probability 1.
k = var("k")
print("sum_{k>=1} 2^-k =", sum(2 ** (-k), k, 1, oo))
partial = [sum(QQ(2) ** (-j) for j in range(1, m + 1)) for m in [1, 2, 5, 10, 20]]
print("partial sums for m = 1, 2, 5, 10, 20:", partial)
print("P(no head in the first 20 tosses) = 1 - partial sum =", 1 - partial[-1], "= 2^-20:", 1 - partial[-1] == QQ(2) ** (-20))
print("expected waiting time sum k 2^-k =", sum(k * 2 ** (-k), k, 1, oo))
