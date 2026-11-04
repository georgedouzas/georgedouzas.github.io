# Exercise exr-fpi-six-point: the 15 pairings of six points, and E X^6 = 15 sigma^6.
P = SetPartitions(6, [2, 2, 2])
print("number of pairings:", P.cardinality())
for p in P:
    print(sorted(sorted(b) for b in p))
x = var("x"); sigma = var("sigma", domain="positive")
print("E X^6 =", (integrate(x ** 6 * exp(-x ** 2 / (2 * sigma ** 2)), x, -oo, oo) / (sigma * sqrt(2 * pi))).simplify_full())
