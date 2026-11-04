# Exercise exr-dq-alkali: hydrogen-like P splitting R alpha^2 Z^4/(n^3 l(l+1)) in cm^-1, checked for hydrogen 2P, and the effective Z for the
# 17 cm^-1 splitting of the sodium 3P level (Svanberg, Table 2.1).
Rinf = 109737.31568; alpha = 1 / 137.035999166
Ra2 = Rinf * alpha^2
print("R alpha^2 = %.3f cm^-1" % Ra2)
print("hydrogen 2P splitting: %.4f cm^-1 = %.2f GHz" % (Ra2 / (8 * 2), Ra2 / 16 * 29.9792458))
Zeff4 = 17 * 27 * 2 / Ra2
print("sodium 3P: Z_eff^4 = %.1f, Z_eff = %.2f" % (Zeff4, Zeff4^(1/4)))
