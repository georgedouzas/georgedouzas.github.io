# Exercise exr-geometric-partial-sums: the closed form s_n = (1 - r^(n+1))/(1 - r) of the geometric partial sums,
# verified symbolically, and the limit 1/(1 - r) for |r| < 1; for r = 2 and r = -1 the series diverges.
n, r, k = var("n r k")
s = sum(r ** k, k, 0, n)
print("s_n =", s.simplify_full(), "; equals (1 - r^(n+1))/(1 - r):", bool((s - (1 - r ** (n + 1)) / (1 - r)).simplify_full() == 0))
for rv in [QQ(1) / 2, QQ(-1) / 3, QQ(9) / 10]:
    print("r =", rv, ": sum r^k =", sum(rv ** k, k, 0, oo), "= 1/(1 - r) =", 1 / (1 - rv))
print("r = 2: partial sums", [sum(2 ** j for j in range(m + 1)) for m in range(6)], "(unbounded); r = -1: partial sums", [sum((-1) ** j for j in range(m + 1)) for m in range(6)], "(no limit)")
