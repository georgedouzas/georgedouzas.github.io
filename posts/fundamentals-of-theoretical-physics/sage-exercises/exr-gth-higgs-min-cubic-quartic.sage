# Exercise exr-gth-higgs-min-cubic-quartic: around the vacuum of V = (lam/4)(|phi|^2 - mu^2)^2 with phi = mu + sigma/sqrt2
# the cubic and quartic self-couplings of the Higgs field sigma are fixed by its mass m^2 = lam mu^2 and the vacuum
# value mu: V = m^2 sigma^2/2 + (m^2/(2 sqrt2 mu)) sigma^3 + (m^2/(16 mu^2)) sigma^4.
sig, mu, lam, m = var("sig mu lam m", domain="positive")
V = (lam / 4 * ((mu + sig / sqrt(2))^2 - mu^2)^2).expand()
c = [V.coefficient(sig, k) for k in range(5)]
print("coefficients of sigma^0..sigma^4:", c)
msq = 2 * c[2]
print("[PASS] m_sigma^2 = lam mu^2:", bool(msq == lam * mu^2))
print("[PASS] cubic coupling = m^2/(2 sqrt2 mu), quartic = m^2/(16 mu^2):", bool((c[3] - msq / (2 * sqrt(2) * mu)).simplify_full() == 0) and bool((c[4] - msq / (16 * mu^2)).simplify_full() == 0))
print("[PASS] the real-displacement form V(mu + h) = lam mu^2 h^2 + lam mu h^3 + lam h^4/4 is the same polynomial with h = sigma/sqrt2:", bool((V - (lam * mu^2 * (sig / sqrt(2))^2 + lam * mu * (sig / sqrt(2))^3 + lam / 4 * (sig / sqrt(2))^4)).expand() == 0))
