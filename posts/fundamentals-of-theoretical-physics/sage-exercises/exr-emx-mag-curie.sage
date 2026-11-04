# Exercise exr-emx-mag-curie: Curie susceptibility of a paramagnetic salt (SI).
mu0 = 4 * pi * 1e-7; kB = 1.380649e-23; muB = 9.2740100783e-24
N = 1e28; mu = 5 * muB
for T in [300, 4]:
    print('T =', T, 'K: chi_m =', n(mu0 * N * mu^2 / (3 * kB * T), digits=3))
x = mu * 1 / (kB * 4)
print('mu B / kT at 1 T and 4 K =', n(x, digits=3))
L = coth(x) - 1 / x
print('Langevin L(x) =', n(L, digits=3), 'versus linear x/3 =', n(x / 3, digits=3))
print('M (Langevin) =', n(N * mu * L, digits=3), 'A/m')
