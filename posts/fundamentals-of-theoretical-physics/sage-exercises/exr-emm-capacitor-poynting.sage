# Exercise exr-emm-capacitor-poynting: energy balance inside a capacitor with circular plates charged by a constant current.
t, s, r, I, a, d = var('t s r I a d', domain='positive')
E = I * t / (pi * a^2)
Bf = lambda rr: I * rr / (2 * pi * a^2)
print('B from the Ampere-Maxwell law (2 pi s B = pi s^2 dE/dt):', bool(2 * pi * s * Bf(s) == pi * s^2 * diff(E, t)))
u = (E^2 + Bf(s)^2) / 2; Ss = -E * Bf(s)          # S = E e_z x B phi-hat = -E B s-hat
print('du/dt + (1/s) d(s S_s)/ds =', (diff(u, t) + diff(s * Ss, s) / s).simplify_full())
flux_in = (-2 * pi * s * d * Ss).simplify_full()
energy = integrate((E^2 + Bf(r)^2) / 2 * 2 * pi * r * d, r, 0, s)
print('inward flux through radius s:', flux_in, '; d/dt of the energy inside:', diff(energy, t).simplify_full())
