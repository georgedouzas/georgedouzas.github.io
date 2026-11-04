# Exercise exr-sl-hermite-form: the multiplier that brings a u'' + b u' + c u = lambda u into Sturm-Liouville form,
# applied to Hermite's equation u'' - 2x u' = -lambda u (p = w = e^(-x^2)).
x, lam = var("x lambda_")
u = function("u")(x)
a, b, c = 1, -2 * x, 0
p = exp(integrate(b / a, x))
mu = p / a
print("p =", p, "| multiplier mu = p/a =", mu, "| weight w = mu =", mu)
lhs = -diff(p * diff(u, x), x) - mu * c * u                 # Sturm-Liouville left side
original = mu * (a * diff(u, x, 2) + b * diff(u, x) + c * u)
print("-(p u')' - mu c u equals -mu (a u'' + b u' + c u):", bool((lhs + original).simplify_full() == 0))
H3 = 8 * x ** 3 - 12 * x                                    # the Hermite polynomial H_3, eigenvalue lambda = 2n = 6
print("H_3 = 8x^3 - 12x satisfies -(e^(-x^2) u')' = 6 e^(-x^2) u:", bool((-diff(p * diff(H3, x), x) - 6 * p * H3).simplify_full() == 0))
print("orthogonality of H_1 = 2x and H_3 with weight e^(-x^2):", integrate(2 * x * H3 * p, x, -oo, oo))
