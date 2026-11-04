# Exercise exr-limit-epsilon-N: for a_n = (2n + 1)/(n + 3) -> 2, find for each epsilon the first N after which
# |a_n - 2| < epsilon for all n >= N, by solving the inequality exactly and by checking terms.
n = var("n")
a = (2 * n + 1) / (n + 3)
print("limit:", limit(a, n=oo), "; |a_n - 2| simplifies to", abs(a - 2).simplify_full(), "= 5/(n + 3)")
for eps in [QQ(1) / 10, QQ(1) / 100, QQ(1) / 1000]:
    N = ceil(5 / eps - 3) + 1
    ok = all(abs(a.subs(n=k) - 2) < eps for k in range(N, N + 50)) and not abs(a.subs(n=N - 1) - 2) < eps
    print("epsilon =", eps, ": N =", N, "(5/(n+3) < eps iff n > 5/eps - 3); checked:", ok)
