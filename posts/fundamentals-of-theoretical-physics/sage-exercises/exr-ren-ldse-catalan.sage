# Exercise exr-ren-ldse-catalan: the anomalous dimension (sqrt(1 + 4a) - 1)/2 of the linear multiedge equation and the
# Catalan numbers.
var('a')
g = taylor((sqrt(1 + 4 * a) - 1) / 2, a, 0, 8)
print("coefficients:", [g.coefficient(a, n) for n in range(1, 9)])
print("(-1)^(n+1) C_(n-1):", [(-1)^(n + 1) * catalan_number(n - 1) for n in range(1, 9)])
print("g(1 + g) - a to order a^8:", taylor(g * (1 + g) - a, a, 0, 8))
print("ratio C_n / C_(n-1) for n = 50:", (catalan_number(50) / catalan_number(49)).n(), " -> 4, radius of convergence 1/4")
