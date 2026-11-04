# Exercise exr-zeta-values: values of the Riemann zeta function at even positive integers, at 0 and at negative
# integers, and the Bernoulli formula for zeta(2n).
for s in (2, 4, 6, 0, -1, -3, -2):
    print("zeta(%d) = %s" % (s, zeta(s)))
for n in (1, 2, 3):
    formula = (-1)^(n + 1)*bernoulli(2*n)*(2*pi)^(2*n)/(2*factorial(2*n))
    print("n = %d: Bernoulli formula gives %s, zeta(2n) = %s, equal: %s" % (n, formula, zeta(2*n), bool(formula == zeta(2*n))))
