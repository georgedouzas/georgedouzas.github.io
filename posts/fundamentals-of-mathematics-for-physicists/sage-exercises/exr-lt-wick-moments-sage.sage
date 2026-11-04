# Exercise exr-lt-wick-moments-sage: Gaussian moments E X^{2n} = (2n-1)!! sigma^{2n} for n <= 6 by symbolic integration,
# the recursion E X^{2n} = (2n-1) sigma^2 E X^{2n-2}, and the count of pairings of 2n points.
x, sigma = var("x sigma")
assume(sigma > 0)
rho = exp(-x ** 2 / (2 * sigma ** 2)) / (sqrt(2 * pi) * sigma)
prev = 1
for n in range(1, 7):
    m = integrate(x ** (2 * n) * rho, x, -oo, oo).simplify_full()
    dfact = (2 * n - 1).multifactorial(2)
    pairings = SetPartitions(2 * n, [2] * n).cardinality()
    print("n = %d: E X^%2d = %s ; (2n-1)!! = %d ; recursion (2n-1) sigma^2 E X^{2n-2} holds: %s ; pairings of %d points: %d" % (n, 2 * n, m, dfact, bool((m - (2 * n - 1) * sigma ** 2 * prev).simplify_full() == 0), 2 * n, pairings))
    prev = m
print("odd moments E X^1, E X^3, E X^5:", [integrate(x ** k * rho, x, -oo, oo) for k in [1, 3, 5]])
