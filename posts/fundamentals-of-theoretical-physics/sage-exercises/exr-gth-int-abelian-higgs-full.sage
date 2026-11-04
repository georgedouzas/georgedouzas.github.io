# Exercise exr-gth-int-abelian-higgs-full: the abelian Higgs model L = -F^2/4 + |D phi|^2 - (lam/4)(|phi|^2 - mu^2)^2,
# D = d + i e A, in unitary gauge phi = mu + sigma/sqrt2. Read off m_sigma, m_A, the couplings sigma A A and
# sigma sigma A A, and the scalar self-couplings, and express the gauge couplings through the masses.
sig, mu, lam, e, AA = var("sig mu lam e AA", domain="positive")      # AA stands for A_mu A^mu
kin = (e^2 * (mu + sig / sqrt(2))^2 * AA).expand()                   # the A-dependent part of |D phi|^2 in unitary gauge
V = (lam / 4 * ((mu + sig / sqrt(2))^2 - mu^2)^2).expand()
mA2 = 2 * kin.coefficient(sig, 0).coefficient(AA); ms2 = 2 * V.coefficient(sig, 2)
print("m_A^2 =", mA2, "; m_sigma^2 =", ms2)
g3 = kin.coefficient(sig, 1).coefficient(AA); g4 = kin.coefficient(sig, 2).coefficient(AA)
print("sigma A A coupling:", g3, "; sigma sigma A A coupling:", g4)
print("[PASS] m_A^2 = 2 e^2 mu^2 and m_sigma^2 = lam mu^2:", bool(mA2 == 2 * e^2 * mu^2) and bool(ms2 == lam * mu^2))
print("[PASS] the sigma A A coupling is m_A^2/(sqrt2 mu) and the sigma sigma A A coupling m_A^2/(4 mu^2):", bool((g3 - mA2 / (sqrt(2) * mu)).simplify_full() == 0) and bool((g4 - mA2 / (4 * mu^2)).simplify_full() == 0))
print("[PASS] m_sigma/m_A = sqrt(lam/2)/e:", bool((sqrt(ms2) / sqrt(mA2) - sqrt(lam / 2) / e).simplify_full() == 0))
print("scalar self-couplings: sigma^3:", V.coefficient(sig, 3), "; sigma^4:", V.coefficient(sig, 4))
print("degrees of freedom: 2 + 2 =", 2 + 2, "before; 1 + 3 =", 1 + 3, "after")
