# Exercise exr-ren-ad-fourier: the Fourier transform of (p^2)^-a in d dimensions, Gamma(d/2 - a)/(4^a pi^(d/2) Gamma(a))
# |x|^(2a - d), checked for d = 3, a = 1 (the Coulomb potential) and the exponent for a = 1 - eta/2.
var('d a eta')
C = lambda dd, aa: gamma(dd / 2 - aa) / (4^aa * pi^(dd / 2) * gamma(aa))
print("d = 3, a = 1: coefficient =", C(3, 1).simplify_full(), " (1/(4 pi) expected)")
print("exponent of |x| for a = 1 - eta/2:", (2 * (1 - eta / 2) - d).expand(), "= -(d - 2 + eta)")
# numerical check in d = 3 of the radial Fourier integral for a = 3/4: (1/(2 pi^2 r)) int_0^oo p sin(p r) p^(-3/2) dp
r = 1.7
num = 1 / (2 * pi^2 * r) * numerical_integral(lambda p: sin(p * r) * p^(-1 / 2), 0, 2000)[0]
print("d = 3, a = 3/4 at r = 1.7: radial integral %.4f (oscillatory tail truncated), formula %.4f" % (num, (C(3, 3 / 4) * r^(2 * 3 / 4 - 3)).n()))
