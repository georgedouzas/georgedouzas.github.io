# Exercise exr-ren-vp-lamb: the Uehling shift of hydrogen nS levels, -4 alpha^5 m_e/(15 pi n^3), in eV and MHz.
al = 1 / 137.035999; me = 510998.95; h = 4.135667696e-15      # eV, eV s
for n in [1, 2]:
    dE = 4 * al^5 * me / (15 * pi * n^3)
    print("n = %d: Delta E = -%.4e eV = -%.2f MHz" % (n, dE.n(), (dE / h / 1e6).n()))
print("2S Uehling shift as a fraction of the 2S-2P Lamb shift 1057.8 MHz: %.2f%%" % (100 * (4 * al^5 * me / (15 * pi * 8) / h / 1e6).n() / 1057.8))
