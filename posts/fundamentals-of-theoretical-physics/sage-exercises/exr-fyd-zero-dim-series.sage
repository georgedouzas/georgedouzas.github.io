# Exercise exr-fyd-zero-dim-series: the zero-dimensional quartic theory Z(lambda) = int dphi/sqrt(2 pi)
# exp(-phi^2/2 - lambda phi^4/24): coefficients c_n = (-1/24)^n (4n-1)!!/n! and partial sums against numerical values.
c = [(-QQ(1) / 24) ** n * prod(range(4 * n - 1, 0, -2)) / factorial(n) for n in range(7)]
print("c_n:", c)
def Znum(l):
    return numerical_integral(lambda p: exp(-p ** 2 / 2 - l * p ** 4 / 24) / sqrt(2 * pi), -20, 20)[0]
for l in [0.1, 1.0]:
    print("lambda = %.1f  Z = %.10f" % (l, Znum(l)))
    print("   partial sums:", ["%.10f" % sum(c[n] * l ** n for n in range(N + 1)) for N in range(7)])
