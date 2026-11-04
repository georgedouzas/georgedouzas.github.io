# Exercise exr-orth-gram-schmidt: Gram-Schmidt on 1, x, x^2, x^3 for the inner product int_{-1}^1 f g dx gives multiples
# of the Legendre polynomials.
x = var("x")
ip = lambda f, g: integrate(f * g, x, -1, 1)
basis = []
for k in range(4):
    e = x ** k - sum(ip(x ** k, b) / ip(b, b) * b for b in basis)
    basis.append(e.expand())
for k, e in enumerate(basis):
    ratio = (e / legendre_P(k, x)).simplify_full()
    print("e_%d =" % k, e, "=", ratio, "* P_%d" % k, "| orthogonal to the previous ones:", all(bool(ip(e, b) == 0) for b in basis[:k]))
