# Exercise exr-rv-cf-moments-sage: moments from derivatives of the characteristic function: a fair die, phi(t) =
# (1/6) sum_k e^{ikt}, has phi^(n)(0)/i^n = E X^n; the uniform variable on [0, 1] has (e^{it} - 1)/(it) with series
# coefficients giving E U^n = 1/(n + 1).
t = var("t")
phi = sum(exp(I * k * t) for k in range(1, 7)) / 6
for n in range(1, 5):
    m = (diff(phi, t, n).subs(t=0) / I ** n).simplify_full()
    direct = sum(QQ(k) ** n for k in range(1, 7)) / 6
    print("n =", n, ": phi^(n)(0)/i^n =", m, ", E X^n =", direct, ":", bool(m == direct))
phiU = (exp(I * t) - 1) / (I * t)
ser = taylor(phiU, t, 0, 5)
for n in range(1, 6):
    coeff = ser.coefficient(t, n) * factorial(n) / I ** n
    print("uniform: E U^%d from the series =" % n, coeff.simplify_full(), "= 1/(n + 1):", bool(coeff.simplify_full() == QQ(1) / (n + 1)))
