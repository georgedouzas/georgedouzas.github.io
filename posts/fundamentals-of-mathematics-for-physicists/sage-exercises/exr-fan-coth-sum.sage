# Exercise exr-fan-coth-sum: sum_{k in Z} 1/(1 + k^2) = pi coth(pi), and the Sobolev constant C_{1,1} on the circle.
k = var("k")
S = 1 + 2 * sum(1 / (1 + k ^ 2), k, 1, oo)
print("sum over Z of 1/(1 + k^2) =", S.simplify_full(), "=", S.n(digits=12))
print("pi coth(pi) =", (pi * coth(pi)).n(digits=12))
x = var("x")
for kk in [0, 1, 3]:
    ck = numerical_integral(cosh(x - pi) * cos(kk * x), 0, 2 * pi)[0] / (2 * pi.n())
    print("k = %d: Fourier coefficient of cosh(x - pi) = %.10f, sinh(pi)/(pi (1 + k^2)) = %.10f" % (kk, ck, (sinh(pi) / (pi * (1 + kk ^ 2))).n()))
C = sqrt(coth(pi) / 2)
print("C_{1,1} = sqrt(coth(pi)/2) =", C.n(digits=6))
