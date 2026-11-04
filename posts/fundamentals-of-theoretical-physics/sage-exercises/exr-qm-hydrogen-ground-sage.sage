# Exercise exr-qm-hydrogen-ground-sage: moments of the hydrogen ground state with hbar, m, k symbolic.
r = var("r", domain="positive"); hbar, m, k = var("hbar m k", domain="positive")
a = hbar ** 2 / (m * k)
C = 1 / sqrt(integrate(exp(-2 * r / a) * 4 * pi * r ** 2, r, 0, oo))
rho = C ** 2 * exp(-2 * r / a) * 4 * pi * r ** 2
E = lambda g: integrate(g * rho, r, 0, oo).simplify_full()
print("C =", C.simplify_full(), "; <r> / a =", (E(r) / a).simplify_full(), "; <r^2> / a^2 =", (E(r ** 2) / a ** 2).simplify_full(), "; <1/r> a =", (E(1 / r) * a).simplify_full())
psi = exp(-r / a)
Hpsi = -hbar ** 2 / (2 * m) * diff(r ** 2 * diff(psi, r), r) / r ** 2 - k / r * psi
E1 = (Hpsi / psi).simplify_full()
print("H psi / psi = E_1 =", E1, "; <V> =", (-k * E(1 / r)).simplify_full(), "; <T> = E_1 - <V> =", (E1 + k * E(1 / r)).simplify_full())
pr = integrate(rho, r, 2 * a, oo).simplify_full()
print("Prob(r > 2a) =", pr, "=", pr.n(digits=5))
