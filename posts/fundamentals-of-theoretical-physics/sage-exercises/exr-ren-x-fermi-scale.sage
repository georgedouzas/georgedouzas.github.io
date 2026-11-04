# Exercise exr-ren-x-fermi-scale: M_W from G_F/sqrt(2) = g^2/(8 M_W^2), and the size of the next operator in muon decay.
GF, g, mmu, MWexp = 1.1664e-5, 0.65, 0.10566, 80.4
MW = sqrt(sqrt(2) * g^2 / (8 * GF))
print("M_W = %.2f GeV (measured %.1f GeV)" % (MW.n(), MWexp))
print("m_mu^2/M_W^2 = %.2e" % (mmu^2 / MWexp^2))
