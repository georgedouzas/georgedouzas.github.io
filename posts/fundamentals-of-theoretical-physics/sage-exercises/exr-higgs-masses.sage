# Exercise exr-higgs-masses: the spectrum of the abelian Higgs model in unitary gauge.
sig, mu, lam, e = var('sig mu lam e', domain='real')
V = lam/4*((mu + sig/sqrt(2))^2 - mu^2)^2
print("V(mu + sigma/sqrt2) =", V.expand())
print("Higgs mass^2 (twice the sigma^2 coefficient):", (2*V.expand().coefficient(sig, 2)).simplify_full())
# gauge boson: |D phi|^2 with D = d + i e A on phi = mu gives e^2 mu^2 A^2 = (1/2) m_A^2 A^2
print("gauge boson mass^2 m_A^2 = 2 e^2 mu^2 ->", 2*e^2*mu^2)
print("degrees of freedom: before (2 real scalars + 2 photon polarisations) =", 2 + 2, "; after (1 Higgs + 3 massive-vector polarisations) =", 1 + 3)
