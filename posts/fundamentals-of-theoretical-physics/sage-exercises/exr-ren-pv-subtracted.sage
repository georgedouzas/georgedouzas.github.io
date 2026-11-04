# Exercise exr-ren-pv-subtracted: the integral of log(1 + x(1 - x) a) over [0, 1], its closed form and its limits.
var('x'); var('a', domain='positive')
F = lambda av: numerical_integral(log(1 + x * (1 - x) * av), 0, 1)[0]
b = lambda av: sqrt(1 + 4 / av)
closed = lambda av: (b(av) * log((b(av) + 1) / (b(av) - 1)) - 2).n()
for av in [0.1, 1, 10, 1000]:
    print("a = %-6s numerical %.8f   closed form %.8f" % (av, F(av), closed(av)))
print("small a:", taylor(integrate(taylor(log(1 + x * (1 - x) * a), a, 0, 3), x, 0, 1), a, 0, 2))
for av in [1e3, 1e5, 1e7]:
    print("large a = %.0e: integral - (log a - 2) = %.2e" % (av, F(av) - (log(av) - 2)))
