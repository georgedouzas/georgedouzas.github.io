# Exercise exr-fan-lp-functional-norm: the functional phi(f) = int_0^1 x^(-1/4) f dx on L^2(0, 1).
x = var('x'); assume(x > 0)
g = x^(-1 / 4)
nrm = sqrt(integrate(g^2, x, 0, 1))
print("|phi| = |x^(-1/4)|_2 =", nrm)
f0 = g / nrm
print("attained at f = x^(-1/4)/sqrt(2): |f|_2 =", sqrt(integrate(f0^2, x, 0, 1)), ", phi(f) =", integrate(g * f0, x, 0, 1))
for f in (1, x):
    print("f = %s: phi(f) = %s, |phi| |f|_2 = %s" % (f, integrate(g * f, x, 0, 1), (nrm * sqrt(integrate(f^2, x, 0, 1))).n(digits=5)))
