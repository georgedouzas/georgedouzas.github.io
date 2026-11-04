# Exercise exr-gr-lambda-value: the observed cosmological constant Lambda = 1.1e-52 m^-2 as an energy density
# rho_Lambda c^2 = Lambda c^4/(8 pi G), in J/m^3 and in natural units, compared with the Planck scale.
Lam, c, G, hbarc, eV = 1.1e-52, 2.99792458e8, 6.674e-11, 1.97327e-7, 1.602177e-19   # hbar c in eV m
rho = Lam*c^4/(8*pi.n()*G)
print("rho_Lambda c^2 = %.2e J/m^3 = %.2e kg/m^3 times c^2" % (rho, rho/c^2))
rho_nat = rho/eV*hbarc^3                                   # eV^4
print("in natural units: %.2e eV^4 = (%.2e eV)^4" % (rho_nat, rho_nat^(1/4)))
MP = 2.435e27                                              # reduced Planck mass (8 pi G)^{-1/2} in eV
print("ratio to M_P^4 (reduced Planck mass): %.1e" % (rho_nat/MP^4))
