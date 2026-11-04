# Exercise exr-power-series-log: integrating the geometric series term by term gives the series of log(1 + x), and
# differentiating it gives 1/(1 + x); Sage's taylor agrees, and the series converges for |x| < 1.
x, n = var("x n")
series_log = sum((-1) ** (n + 1) * x ** n / n, n, 1, oo)
print("sum (-1)^(n+1) x^n/n =", series_log, "for |x| < 1")
print("taylor of log(1 + x) to order 6:", taylor(log(1 + x), x, 0, 6))
print("derivative of the series equals the geometric series 1/(1 + x):", bool((diff(series_log, x) - 1 / (1 + x)).simplify_full() == 0))
print("at x = 1 the series is the alternating harmonic series, sum = log 2:", sum((-1) ** (n + 1) / n, n, 1, oo))
