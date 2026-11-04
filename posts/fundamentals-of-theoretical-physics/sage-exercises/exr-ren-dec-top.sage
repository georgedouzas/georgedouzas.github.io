# Exercise exr-ren-dec-top: one-loop alpha_s from M_Z to 1 TeV with five flavours, and with the top threshold at 172.5 GeV.
MZ, mt, mu = 91.1876, 172.5, 1000.0
inv5 = 1 / 0.118 + (23 / 3) / (2 * pi) * log(mu / MZ)
inv_t = 1 / 0.118 + (23 / 3) / (2 * pi) * log(mt / MZ)
inv6 = inv_t + 7 / (2 * pi) * log(mu / mt)
print("five flavours: 1/alpha_s(1 TeV) = %.3f, alpha_s = %.4f" % (inv5.n(), (1 / inv5).n()))
print("with threshold: 1/alpha_s(m_t) = %.3f, 1/alpha_s(1 TeV) = %.3f, alpha_s = %.4f (%.1f%% larger)" % (inv_t.n(), inv6.n(), (1 / inv6).n(), (100 * (inv5 / inv6 - 1)).n()))
