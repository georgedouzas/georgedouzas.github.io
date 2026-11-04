# Exercise exr-rn-pion-rate: the anomaly prediction Gamma(pi0 -> gamma gamma) = alpha^2 m^3/(64 pi^3 f^2) and the
# lifetime; the rate scales as N_c^2 (Q_u^2 - Q_d^2)^2.
alpha = 1 / 137.036; m = 134.98; f = 92.1; hbar = 6.582e-22
G = alpha ** 2 * m ** 3 / (64 * RDF(pi) ** 3 * f ** 2)      # MeV
print("Gamma = %.3f eV,  tau = %.3e s" % (G * 1e6, hbar / G))
for Nc in [1, 3]:
    print("N_c = %d: N_c^2 (Q_u^2 - Q_d^2)^2 =" % Nc, Nc ** 2 * (QQ(4) / 9 - QQ(1) / 9) ** 2)
