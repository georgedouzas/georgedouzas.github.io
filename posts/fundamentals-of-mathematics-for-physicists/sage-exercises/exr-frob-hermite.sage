# Exercise exr-frob-hermite (Cini et al., Secs. 7.1.3 and 7.2): series solutions of Hermite's equation
# y'' - 2x y' + 2 nu y = 0 about the ordinary point 0, a_{j+2} = 2(j - nu)/((j + 1)(j + 2)) a_j. For integer nu one
# series terminates (a multiple of the Hermite polynomial H_nu); otherwise the even series grows like
# x^(-nu-1) e^(x^2), and e^(-x^2/2) y is not square integrable.
x = var("x")
def series(nu, parity, terms):
    a = {parity: 1}
    for j in range(parity, terms, 2):
        a[j + 2] = 2*(j - nu)/((j + 1)*(j + 2))*a[j]
    return sum(c*x^j for j, c in a.items())
for nu in [0, 1, 2, 3, 4]:
    y = series(nu, nu % 2, 30)
    print("nu = %d: terminating series %s, proportional to H_%d: %s" % (nu, y, nu, bool((y/hermite(nu, x)).simplify_full().is_constant())))
nu = 1/2
y = series(nu, 0, 400)
print("nu = 1/2: the truncated even series leaves a residual of degree", (diff(y, x, 2) - 2*x*diff(y, x) + 2*nu*y).expand().degree(x), "only")
for X in [2, 4, 6, 8]:
    print("x = %d: y x^(3/2) e^(-x^2) = %.4f, e^(-x^2/2) y = %.4g" % (X, (y(x=X)*X^(3/2)/exp(X^2)).n(), (exp(-X^2/2)*y(x=X)).n()))
