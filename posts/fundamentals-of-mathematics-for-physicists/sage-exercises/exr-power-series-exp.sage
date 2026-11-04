# Exercise exr-power-series-exp: the Taylor polynomials of exp, sin and 1/(1 - x) at 0 agree with the first terms
# of their power series, and the error of the degree-n polynomial of exp at x = 1 is below 1/n!.
x = var("x")
print("exp:", taylor(exp(x), x, 0, 5))
print("sin:", taylor(sin(x), x, 0, 7))
print("1/(1-x):", taylor(1 / (1 - x), x, 0, 5))
for m in [3, 6, 9]:
    T = sum(QQ(1) / factorial(k) for k in range(m + 1))
    print("degree", m, "polynomial of exp at x = 1:", numerical_approx(T, digits=10), "; |e - T| =", numerical_approx(abs(e - T), digits=4), "< 1/", m, "! =", numerical_approx(1 / factorial(m), digits=4), ":", bool(abs(e - T) < 1 / factorial(m)))
