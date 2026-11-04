# Exercise exr-rn-qed-running: one-loop running of alpha from the electron mass to the Z mass,
# 1/alpha(mu) = 1/alpha(m_e) - (2/(3 pi)) sum_leptons log(mu/m_l) (each lepton from its own mass).
a0 = 137.036; mZ = 91.19
masses = {"e": 0.000511, "mu": 0.1057, "tau": 1.777}
inv_e = a0 - 2 / (3 * RDF(pi)) * log(mZ / masses["e"])
inv_all = a0 - sum(2 / (3 * RDF(pi)) * log(mZ / mm) for mm in masses.values())
print("1/alpha(m_Z), electron only: %.2f" % inv_e)
print("1/alpha(m_Z), e, mu, tau:    %.2f" % inv_all)
