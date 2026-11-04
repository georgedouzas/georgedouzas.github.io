# Exercise exr-ren-coleman-weinberg: zeta(s) = V m^(4 - 2s)/(16 pi^2 (s - 1)(s - 2)) for -Delta + m^2 in four dimensions,
# the renormalised effective action -zeta'(0)/2 - log(mu^2) zeta(0)/2, and the Coleman-Weinberg potential.
var('s'); var('m mu', domain='positive')
print("Gamma(s - 2)/Gamma(s) =", (gamma(s - 2) / gamma(s)).simplify_full().factor())
zeta_D = m^(4 - 2 * s) / (16 * pi^2 * (s - 1) * (s - 2))
z0 = zeta_D.subs(s=0); dz0 = diff(zeta_D, s).subs(s=0)
W = -dz0 / 2 - log(mu^2) * z0 / 2
print("zeta(0)/V =", z0, "   zeta'(0)/V =", dz0.simplify_log())
print("W/V equals m^4/(64 pi^2) (log(m^2/mu^2) - 3/2):", bool((W - m^4 / (64 * pi^2) * (log(m^2 / mu^2) - 3 / 2)).simplify_log().simplify_full() == 0))
var('phi lam')
M2 = m^2 + lam * phi^2 / 2
print("V_eff =", m^2 * phi^2 / 2 + lam * phi^4 / 24 + M2^2 / (64 * pi^2) * (log(M2 / mu^2) - 3 / 2))
