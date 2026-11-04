# Exercise exr-ren-lp-alpha-s: one-loop running of alpha_s from M_Z to m_b (n_f = 5) and to m_tau (n_f = 4).
MZ, mb, mtau = 91.1876, 4.18, 1.777
inv_b = 1 / 0.118 + (23 / 3) / (2 * pi) * log(mb / MZ)
inv_tau = inv_b + (25 / 3) / (2 * pi) * log(mtau / mb)
print("1/alpha_s(m_b) = %.3f, alpha_s(m_b) = %.4f" % (inv_b.n(), (1 / inv_b).n()))
print("1/alpha_s(m_tau) = %.3f, alpha_s(m_tau) = %.4f" % (inv_tau.n(), (1 / inv_tau).n()))
