# Exercise exr-gr-radiation-matter: a flat universe with radiation and dust. In conformal time (dt = a d eta) the first
# Friedmann equation reads (da/deta)^2 = C_r + C_m a with C_r = (8 pi G/3) rho_r a^4 and C_m = (8 pi G/3) rho_m a^3
# constant; a = sqrt(C_r) eta + C_m eta^2/4 solves it, radiation-like (a ~ eta ~ t^{1/2}) early and matter-like late.
eta, Cr, Cm = var('eta Cr Cm'); assume(Cr > 0, Cm > 0, eta > 0)
a = sqrt(Cr)*eta + Cm*eta^2/4
print("(a')^2 - C_r - C_m a =", (diff(a, eta)^2 - Cr - Cm*a).expand())
t = integrate(a, eta)
print("t(eta) =", t, "; at small eta t ~ sqrt(C_r) eta^2/2, so a ~ (2 sqrt(C_r) t)^{1/2}; at large eta t ~ C_m eta^3/12, so a ~ t^{2/3}")
eq_eta = 2*(sqrt(2) - 1)*sqrt(Cr)/Cm
print("equality (rho_r = rho_m, a = C_r/C_m) at eta = 2(sqrt 2 - 1) sqrt(C_r)/C_m:", bool((a.subs(eta=eq_eta) - Cr/Cm).simplify_full() == 0))
