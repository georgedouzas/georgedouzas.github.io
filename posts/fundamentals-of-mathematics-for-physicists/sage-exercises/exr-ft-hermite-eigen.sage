# Exercise exr-ft-hermite-eigen: the first Hermite functions are eigenfunctions of the Fourier transform with
# eigenvalues sqrt(2 pi) (-i)^n.
x, k = var("x k")
assume(k, "real")
F = lambda f: integrate(f * exp(-I * k * x), x, -oo, oo).simplify_full()
for n, h in enumerate([exp(-x ** 2 / 2), x * exp(-x ** 2 / 2), (2 * x ** 2 - 1) * exp(-x ** 2 / 2)]):
    lam = sqrt(2 * pi) * (-I) ** n
    print("n =", n, ": transform =", F(h), "; equals sqrt(2 pi) (-i)^n h(k):", bool((F(h) - lam * h.subs(x=k)).simplify_full() == 0))
