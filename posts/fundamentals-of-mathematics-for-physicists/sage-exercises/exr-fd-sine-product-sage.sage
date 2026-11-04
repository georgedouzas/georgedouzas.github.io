# Exercise exr-fd-sine-product-sage: Euler's product prod (1 - z/(n pi)^2) against sin(sqrt z)/sqrt z, and tr G = 1/6,
# tr G^2 = 1/90 from the expansion of log(sin(sqrt z)/sqrt z).
for zz in [1, pi ** 2 / 4]:
    exact = (sin(sqrt(zz)) / sqrt(zz)).n(digits=10)
    print("z =", zz, ": sin(sqrt z)/sqrt z =", exact, "; partial products N = 10, 100, 1000:", [prod(1 - RR(zz) / (n * RR(pi)) ** 2 for n in range(1, N + 1)) for N in [10, 100, 1000]])
z = var("z")
print("sin(sqrt z)/sqrt z =", taylor(sin(sqrt(z)) / sqrt(z), z, 0, 2))
L = taylor(log(sin(sqrt(z)) / sqrt(z)), z, 0, 2)
print("log(sin(sqrt z)/sqrt z) =", L, "-> tr G =", -L.coefficient(z, 1), ", tr G^2 =", -2 * L.coefficient(z, 2))
