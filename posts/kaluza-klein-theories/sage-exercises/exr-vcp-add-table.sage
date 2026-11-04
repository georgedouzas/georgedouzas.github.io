# Exercise exr-vcp-add-table: the size R of n extra dimensions from M_Pl^2 = M^{n+2} R^n with M_Pl = 1.22e19 GeV and
# M = 1 TeV, the KK spacing 1/R in eV, and the torus convention M_Pl^2 = M^{n+2} (2 pi R_t)^n, which divides R by 2 pi.
MPl, M, hbarc = 1.22e19, 1e3, 1.97327e-16
for n in range(1, 8):
    R = (MPl**2 / M**(n + 2))**(1.0 / n)          # GeV^-1
    print("n = %d: R = %.2e m, 1/R = %.2e eV, torus radius R_t = R/(2 pi) = %.2e m" % (n, R * hbarc, 1e9 / R, R * hbarc / (2 * pi.n())))
