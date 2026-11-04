# Exercise exr-fs-parseval-square: Parseval's identity for the square wave, (1/pi) int sgn(x)^2 = 2 = sum over odd n of
# (4/(pi n))^2, giving sum_{odd} 1/n^2 = pi^2/8 and sum 1/n^2 = pi^2/6.
n = var("n")
lhs = (integrate(1, x, -pi, pi)) / pi
rhs = sum((4 / (pi * (2 * n + 1))) ** 2, n, 0, oo)
print("(1/pi) int sgn^2 =", lhs, "; sum_{odd n} (4/(pi n))^2 =", rhs, "; equal:", bool(lhs == rhs))
odd = sum(1 / (2 * n + 1) ** 2, n, 0, oo)
print("sum over odd n of 1/n^2 =", odd, "= pi^2/8:", bool(odd == pi ** 2 / 8))
print("sum 1/n^2 = (4/3) pi^2/8 =", 4 / 3 * odd, "= pi^2/6:", bool(4 / 3 * odd == pi ** 2 / 6), "; Sage's sum:", sum(1 / n ** 2, n, 1, oo))
