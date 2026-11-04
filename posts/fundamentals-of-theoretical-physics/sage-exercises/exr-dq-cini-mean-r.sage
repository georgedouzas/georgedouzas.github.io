# Exercise exr-dq-cini-mean-r: <r> = (2 gamma + 1)/(2 m Z alpha) for the Dirac ground state (density ~ r^{2 gamma} e^{-2 lambda r}),
# against the Schrodinger 3/(2 m Z alpha); relative difference for Z = 1 and Z = 92.
r, lam, g = var("r lam g", domain="positive")
Mom = lambda s_: gamma(s_ + 1) / (2 * lam)^(s_ + 1)        # int_0^oo r^s e^{-2 lambda r} dr
print("check of the moment formula at s = 1.4, lambda = 0.7:", abs(numerical_integral(r^1.4 * exp(-1.4 * r), 0, oo)[0] - Mom(1.4).subs(lam=0.7).n()) < 1e-8)
mean = (Mom(2 * g + 1) / Mom(2 * g)).simplify_full()
print("<r> =", mean, "; equals (2 gamma + 1)/(2 lambda):", bool((mean - (2 * g + 1) / (2 * lam)).simplify_full() == 0))
alpha = 1 / 137.035999166
for Z in (1, 92):
    gv = sqrt(1 - (Z * alpha)^2)
    print("Z = %d: gamma = %.6f, <r>/a_Z = %.6f, relative difference from 3/2: %.3g" % (Z, gv, (2 * gv + 1) / 2, ((2 * gv + 1) / 2 - 1.5) / 1.5))
