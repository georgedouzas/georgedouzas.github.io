# Exercise exr-acsym-div-two-waves: the U(1) current j^mu = i(phibar d^mu phi - phi d^mu phibar) of a superposition of
# two positive-frequency plane waves on the circle 0 <= x < 2 pi in 1+1 dimensions (signature (+, -)). The density
# has an interference term, but the charge is the sum of the two charges and is constant in time.
t, x = var('t x', domain='real'); m, a, b = var('m a b', domain='positive')
k1, k2 = 1, 2
w1, w2 = sqrt(k1^2 + m^2), sqrt(k2^2 + m^2)
phi = a * exp(-I * (w1 * t - k1 * x)) + b * exp(-I * (w2 * t - k2 * x)); phib = phi.conjugate()
print("phi solves the Klein-Gordon equation:", bool((diff(phi, t, 2) - diff(phi, x, 2) + m^2 * phi).simplify_full() == 0))
j0 = (I * (phib * diff(phi, t) - phi * diff(phib, t))).simplify_full()
j1 = (-I * (phib * diff(phi, x) - phi * diff(phib, x))).simplify_full()  # j^1 = -j_x
interf = 2 * a * b * (w1 + w2) * cos((w2 - w1) * t - (k2 - k1) * x)
print("j^0 = 2(a^2 omega_1 + b^2 omega_2) + 2ab(omega_1 + omega_2) cos((omega_2 - omega_1) t - (k_2 - k_1) x):", bool((j0 - 2 * (a^2 * w1 + b^2 * w2) - interf).expand().simplify_trig() == 0))
print("d_t j^0 + d_x j^1 =", (diff(j0, t) + diff(j1, x)).simplify_full())
Q = integrate(j0.expand(), x, 0, 2 * pi).simplify_full()
print("Q = int_0^{2 pi} j^0 dx =", Q, "  equals 4 pi (a^2 omega_1 + b^2 omega_2):", bool((Q - 4 * pi * (a^2 * w1 + b^2 * w2)).simplify_full() == 0))
