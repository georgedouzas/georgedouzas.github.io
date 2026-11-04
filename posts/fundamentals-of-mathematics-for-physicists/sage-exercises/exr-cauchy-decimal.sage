# Exercise exr-cauchy-decimal: the partial sums s_n = sum_{k<=n} 1/k! form a Cauchy sequence: |s_m - s_n| for m > n
# is bounded by 2/(n+1)! (a geometric tail estimate), so the distances shrink faster than any power of 1/n.
s = lambda m: sum(1 / factorial(k) for k in range(0, m + 1))
for n in [2, 4, 6, 8]:
    d = max(abs(s(m) - s(n)) for m in range(n + 1, n + 20))
    print("n =", n, ": max_{n<m<n+20} |s_m - s_n| =", float(d), "<= 2/(n+1)! =", float(2 / factorial(n + 1)), ":", d <= 2 / factorial(n + 1))
print("the limit is e: |s_12 - e| =", float(abs(s(12) - e)), "< 1e-9:", bool(abs(s(12) - e) < 1e-9))
