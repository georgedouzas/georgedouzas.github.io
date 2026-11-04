# Exercise exr-emx-en-born: field energy outside a charged sphere in a dielectric, and the Born energy (SI numbers).
r, q, a, eps = var('r q a epsilon', domain='positive')
D = q / (4 * pi * r^2)
W = integrate(D^2 / (2 * eps) * 4 * pi * r^2, r, a, oo)
print('W =', W)
e = 1.602176634e-19; eps0 = 8.8541878128e-12
dW = e^2 / (8 * pi * eps0 * 0.2e-9) * (1 - 1 / 80)
print('energy gained entering water:', n(dW / e, digits=3), 'eV')
