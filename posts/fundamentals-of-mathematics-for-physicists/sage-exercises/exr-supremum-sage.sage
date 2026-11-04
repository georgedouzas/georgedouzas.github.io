# Exercise exr-supremum-sage: the set {1 - 1/n : n >= 1} has supremum 1, which is not attained; the first terms
# approach it from below, and the limit of the sequence equals the supremum.
n = var("n")
terms = [1 - QQ(1) / k for k in range(1, 11)]
print("first terms 1 - 1/n:", terms)
print("every term is < 1:", all(t < 1 for t in terms), "; the largest of the first 10 is", max(terms), "< 1")
print("limit of 1 - 1/n:", limit(1 - 1 / n, n=oo), "= the supremum; for epsilon = 1/1000 the term with n = 1001 exceeds 1 - epsilon:", bool(1 - QQ(1) / 1001 > 1 - QQ(1) / 1000))
S = {QQ(1) / k for k in range(1, 11)}
print("the set {1/n} has supremum max = 1 (attained) and infimum 0 (not attained): max =", max(S), ", limit =", limit(1 / n, n=oo))
