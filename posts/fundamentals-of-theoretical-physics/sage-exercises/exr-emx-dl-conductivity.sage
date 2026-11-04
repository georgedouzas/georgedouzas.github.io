# Exercise exr-emx-dl-conductivity: limits of the Drude conductivity and the Drude permittivity.
w, g, wp = var('omega gamma omega_p', domain='positive')
s0 = wp^2 / g
sig = s0 / (1 - I * w / g)
print('omega << gamma:', sig.series(w, 2))
u = var('u')    # u = 1/omega
print('omega >> gamma (leading term):', sig.subs({w: 1 / u}).series(u, 2))
eps = 1 + I * sig / w
print('1 + i sigma/omega = 1 - wp^2/(omega(omega + i gamma)):', bool((eps - (1 - wp^2 / (w * (w + I * g)))).simplify_full() == 0))
