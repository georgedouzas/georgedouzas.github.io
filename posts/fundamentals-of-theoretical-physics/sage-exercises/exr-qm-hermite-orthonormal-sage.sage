# Exercise exr-qm-hermite-orthonormal-sage: Gram matrix of the Hermite functions psi_0..psi_5 and the ladder relation.
xi = var("xi", domain="real")
psi = lambda n: pi ** (-1 / 4) / sqrt(2 ** n * factorial(n)) * hermite(n, xi) * exp(-xi ** 2 / 2)
G = matrix(SR, 6, 6, lambda i, j: integrate(psi(i) * psi(j), xi, -oo, oo).simplify_full())
print("Gram matrix:"); print(G)
a = lambda f: (xi * f + diff(f, xi)) / sqrt(2)
print("a psi_n = sqrt(n) psi_(n-1), n = 1..5:", all(bool((a(psi(n)) - sqrt(n) * psi(n - 1)).canonicalize_radical() == 0) for n in range(1, 6)), "; a psi_0 = 0:", bool(a(psi(0)).simplify_full() == 0))
print("Hermite polynomials:", [hermite(n, xi) for n in range(5)])
