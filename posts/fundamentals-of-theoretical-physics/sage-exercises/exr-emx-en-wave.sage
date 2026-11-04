# Exercise exr-emx-en-wave: plane wave in a linear medium and Poynting's theorem.
z, t, w, E0, eps, mu = var('z t omega E0 epsilon mu', domain='positive')
nn = sqrt(eps * mu)
Ex = E0 * cos(nn * w * z - w * t)
By = nn * Ex
Hy = By / mu; Dx = eps * Ex
print('Faraday  (dz Ex = -dt By):', bool((diff(Ex, z) + diff(By, t)).simplify_full() == 0))
print('Ampere-Maxwell (-dz Hy = dt Dx):', bool((-diff(Hy, z) - diff(Dx, t)).simplify_full() == 0))
u = (Ex * Dx + By * Hy) / 2
S = Ex * Hy
print('du/dt + dS/dz =', (diff(u, t) + diff(S, z)).simplify_full())
print('S/u =', (S / u).simplify_full())
