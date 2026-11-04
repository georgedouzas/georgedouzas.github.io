# Exercise exr-dq-lamb-estimate: the estimates 4 m (Z alpha)^4 alpha ln((Z alpha)^-2)/(3 pi n^3) and -4 m (Z alpha)^4 alpha/(15 pi n^3) of the
# Lamb shift of 2S (n = 2, Z = 1) in MHz, against the measured 1057.845 MHz.
alpha = 1 / 137.035999166; me = 510998.95069; h = 4.135667696e-15   # eV, eV s
ma5 = me * alpha^5 / h / 1e6   # m alpha^5 in MHz
dE = 4 * ma5 * log(alpha^-2) / (3 * pi * 8); dVP = -4 * ma5 / (15 * pi * 8)
print("m alpha^5 = %.1f MHz, ln(alpha^-2) = %.3f" % (ma5, log(alpha^-2)))
print("charge-radius estimate %.0f MHz, vacuum polarisation %.1f MHz, sum %.0f MHz, measured 1057.8 MHz (ratio %.2f)" % (dE.n(), dVP.n(), (dE + dVP).n(), ((dE + dVP) / 1057.845).n()))
