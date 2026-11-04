# Exercise exr-ren-x-casimir-regulators: the zero-point energy (1/2) sum |n|/R on a circle with an exponential cutoff,
# a Gaussian (heat-kernel) cutoff, and the zeta function.
var('d'); var('R', domain='positive')
E_exp = exp(d / R) / (R * (exp(d / R) - 1)^2)
print("exponential cutoff:", E_exp.series(d, 2))
for dv in [0.1, 0.03, 0.01]:
    g = sum(n * exp(-RealField(100)(dv)^2 * n^2) for n in range(1, 4000))
    print("Gaussian cutoff, R = 1, delta = %.2f: sum - 1/(2 delta^2) = %.8f" % (dv, g - 1 / (2 * dv^2)))
print("zeta: zeta(-1) =", zeta(-1))
