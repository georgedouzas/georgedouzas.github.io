# Exercise exr-qmb-swave (Cini et al., Sec. 14.10, Problem 3): s-wave scattering by the repulsive spherical barrier
# V = V0 for r < a (units hbar^2/2m = 1, a = 1). For E < V0: tan(delta_0 + k a) = (k/kappa) tanh(kappa a),
# sigma_0 = 4 pi sin^2(delta_0)/k^2; low-energy limit 4 pi a_s^2 with a_s = a (1 - tanh(kappa_0 a)/(kappa_0 a)),
# and the hard-sphere limit a_s -> a.
k, V0 = var('k V0')
def delta0(kv, V):
    kap = sqrt(V - kv^2)
    return (arctan(kv / kap * tanh(kap)) - kv).n()
for V in (1, 10, 100, 10^4):
    k0 = sqrt(V).n()
    a_s = 1 - tanh(k0) / k0
    kv = 1e-3
    sig = 4 * pi.n() * sin(delta0(kv, V))^2 / kv^2
    print("V0 = %6d: sigma_0(k -> 0) = %.6f, 4 pi a_s^2 = %.6f, a_s = %.5f" % (V, sig, 4 * pi.n() * a_s^2, a_s))
print("hard sphere: delta_0 = -k a, sigma_0 = 4 pi sin^2(ka)/k^2, at ka = 0.5:", (4 * pi * sin(0.5)^2 / 0.25).n(digits=6), "vs geometric pi a^2 =", pi.n(digits=6))
