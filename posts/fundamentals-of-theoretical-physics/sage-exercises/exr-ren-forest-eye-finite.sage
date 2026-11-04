# Exercise exr-ren-forest-eye-finite: the renormalised eye graph (1 - K)(eye - K(B) B) in the modified minimal scheme,
# units (16 pi^2)^-2, d = 4 - eps, L = log(p^2/mu^2).
var('e L'); d = 4 - e
G = lambda a, b: gamma(a + b - d / 2) * gamma(d / 2 - a) * gamma(d / 2 - b) / (gamma(a) * gamma(b) * gamma(d - a - b))
B = exp(euler_gamma * e / 2) * G(1, 1) * exp(-e * L / 2)
eye = exp(euler_gamma * e) * G(1, 1) * G(1 + e / 2, 1) * exp(-e * L)
barR = (eye - (2 / e) * B).series(e, 1).truncate().expand()
poles = sum(barR.coefficient(e, -k) * e^(-k) for k in [1, 2])
print("local poles removed by the overall counterterm:", poles, "  (independent of L:", bool(diff(poles, L) == 0), ")")
print("renormalised eye graph at eps = 0:", (barR - poles).coefficient(e, 0).expand())
