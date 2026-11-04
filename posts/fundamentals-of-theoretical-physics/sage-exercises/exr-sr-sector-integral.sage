# Exercise exr-sr-sector-integral: the Euclidean area of the hyperbolic sector of the unit hyperbola equals half the rapidity.
var('t eta phi', domain='positive')
assume(cosh(eta) - 1 > 0)
A = cosh(eta)*sinh(eta)/2 - integrate(sqrt(t^2 - 1), t, 1, cosh(eta))
print("area =", A.simplify_full())
print("area - eta/2 at eta = 0.3, 1, 2.5:", [n(A.subs(eta=e) - e/2) for e in [0.3, 1, 2.5]])
