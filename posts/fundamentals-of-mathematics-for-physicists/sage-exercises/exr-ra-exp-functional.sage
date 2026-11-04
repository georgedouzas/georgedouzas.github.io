# Exercise exr-ra-exp-functional (integrative): the exponential defined by its power series satisfies E(x + y) =
# E(x) E(y) (Cauchy product of absolutely convergent series) and E' = E (term-by-term differentiation); checked on
# truncations to high order and exactly with Sage's series arithmetic.
x, y = var("x y")
N = 12
E = lambda z: sum(z ** k / factorial(k) for k in range(N + 1))
diff_poly = (E(x + y) - E(x) * E(y)).expand()
low = sum(c * x ** i * y ** j for (c, i, j) in [(diff_poly.coefficient(x, i).coefficient(y, j), i, j) for i in range(N + 1) for j in range(N + 1 - i)])
print("E_N(x + y) - E_N(x) E_N(y) has no terms of total degree <= N:", bool(low == 0), "(the mismatch is only in the truncated tail)")
print("derivative of the truncation: E_N'(x) - E_{N-1}(x) =", (diff(E(x), x) - sum(x ** k / factorial(k) for k in range(N))).expand(), "(equal up to the dropped term)")
print("exact: the Cauchy product coefficient sum_{i+j=n} 1/(i! j!) = 2^n/n! for n = 0..6:", [bool(sum(QQ(1) / (factorial(i) * factorial(n - i)) for i in range(n + 1)) == QQ(2) ** n / factorial(n)) for n in range(7)])
print("so E(1)^2 = sum 2^n/n! = E(2) = e^2:", bool(sum(2 ** var("n") / factorial(var("n")), var("n"), 0, oo) == exp(2)))
