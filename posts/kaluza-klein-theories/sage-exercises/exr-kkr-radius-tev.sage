# Exercise exr-kkr-radius-tev: what a TeV-size circle would imply. With M_P = 1.22e19 GeV (kappa = sqrt(8 pi)/M_P) and
# m_1 = 1/R = 1 TeV, the Kaluza-Klein gauge coupling e = sqrt(2) kappa/R and alpha = e^2/(4 pi) = 4/(M_P R)^2 are
# absurdly small: the photon of a TeV circle cannot be the photon of electromagnetism.
MP = 1.22e19; m1 = 1.0e3
kappa = sqrt(8 * pi.n()) / MP; R = 1 / m1
e = sqrt(2) * kappa / R; alpha = e^2 / (4 * pi.n())
print("R = %.3e GeV^-1 = %.3e m, e = %.3e, alpha = %.3e" % (R, R * 1.9733e-16, e, alpha))
print("[PASS] alpha = 4/(M_P R)^2 for m_1 = 1 TeV, about 2.7e-32:", bool(abs(alpha - 4 / (MP * R)^2) < 1e-40) and bool(2e-32 < alpha < 3e-32))
