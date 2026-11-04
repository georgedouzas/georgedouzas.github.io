# Exercise exr-riesz-sage-coefficients: the functional phi(f) = int_{-1}^1 x^3 f(x) dx on polynomials of degree <= 2 is
# represented by the projection of x^3 onto that subspace, computed in the orthonormal Legendre basis.
x = var("x", domain="real")
ip = lambda f, g: integrate(f * g, x, -1, 1)
es = [sqrt((2 * k + 1) / QQ(2)) * legendre_P(k, x) for k in range(3)]
v = sum(ip(e, x ** 3) * e for e in es).expand().simplify_full()
print("representing vector v = projection of x^3 onto degree <= 2:", v)
for f in [SR(1), x, x ** 2]:
    print("f =", f, ": phi(f) =", ip(x ** 3, f), " <v, f> =", ip(v, f))
