# Exercise exr-dq-klein-sage: R + T = 1 for R = ((1 - kappa)/(1 + kappa))^2, T = 4 kappa/(1 + kappa)^2; the limit of kappa as V -> infinity
# in the Klein zone, and T_infinity for E = 3m and E = 1.1m.
kap = var("kappa"); E, m, V = var("E m V", domain="positive")
R = ((1 - kap) / (1 + kap))^2; T = 4 * kap / (1 + kap)^2
print("R + T = 1:", bool((R + T - 1).simplify_full() == 0))
k = sqrt(E^2 - m^2); q = -sqrt((V - E)^2 - m^2)
kv = q / k * (E + m) / (E - V + m)
lim = limit(kv, V=oo)
print("lim_{V -> oo} kappa =", lim.simplify_full(), "; equals sqrt((E + m)/(E - m)):", bool((lim^2 - (E + m) / (E - m)).simplify_full() == 0))
for Ev in (3, 1.1):
    ki = sqrt((Ev + 1) / (Ev - 1))
    print("E = %s m: kappa_inf = %.4f, T_inf = %.4f" % (Ev, ki.n(), (4 * ki / (1 + ki)^2).n()))
