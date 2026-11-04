# Exercise exr-fan-weak-products: pairings of sin(nx), sin^2(nx), sin(nx)cos(nx) with test functions on (0, 2 pi).
x = var('x'); n = var('n', domain='integer'); assume(n > 0)
print("int x sin^2(nx) =", integrate(x * sin(n * x)^2, x, 0, 2 * pi).simplify_full(), " (= pi^2 = (1/2) int x)")
for g in (x, x^2, exp(-x)):
    vals = [integrate(g * sin(m * x), x, 0, 2 * pi).n(digits=5) for m in (1, 10, 100)]
    vals2 = [(integrate(g * sin(m * x)^2, x, 0, 2 * pi) - integrate(g / 2, x, 0, 2 * pi)).n(digits=5) for m in (1, 10, 100)]
    vals3 = [integrate(g * sin(m * x) * cos(m * x), x, 0, 2 * pi).n(digits=5) for m in (1, 10, 100)]
    print("g = %s: <g, sin nx> %s; <g, sin^2 nx - 1/2> %s; <g, sin nx cos nx> %s (n = 1, 10, 100)" % (g, vals, vals2, vals3))
