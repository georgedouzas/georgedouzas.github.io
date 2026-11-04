# Exercise exr-ra-stirling (integrative): log n! = sum_{k<=n} log k is squeezed between int_1^n log x dx and
# int_1^{n+1} log x dx, giving Stirling's growth n log n - n + O(log n); the ratio n!/(sqrt(2 pi n) (n/e)^n) tends to 1.
x = var("x")
for n in [5, 10, 50, 200]:
    lower = integrate(log(x), x, 1, n); upper = integrate(log(x), x, 1, n + 1)
    L = log(factorial(n))
    ratio = factorial(n) / (sqrt(2 * pi * n) * (n / e) ** n)
    print("n =", n, ": int_1^n log x =", lower.n(digits=6), "<= log n! =", L.n(digits=6), "<= int_1^{n+1} log x =", upper.n(digits=6), ":", bool(lower <= L <= upper), "; Stirling ratio", ratio.n(digits=8))
print("int log x dx = x log x - x, so log n! = n log n - n + O(log n); the O(log n) term is (1/2) log(2 pi n) + o(1)")
