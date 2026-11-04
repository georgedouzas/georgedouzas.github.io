# Exercise exr-emx-mm-electromagnet: field in the gap of an iron ring, B = mu0 N I/(w + l/mu_r) (SI).
mu0 = 4 * pi * 1e-7
N, I, mur = 400, 2, 1000
for w in [1e-3, 5e-3, 10e-3]:
    l = 2 * pi * 0.1 - w
    print('w =', w * 1e3, 'mm: l/mu_r =', n(l / mur * 1e3, digits=3), 'mm, B =', n(mu0 * N * I / (w + l / mur), digits=3), 'T')
