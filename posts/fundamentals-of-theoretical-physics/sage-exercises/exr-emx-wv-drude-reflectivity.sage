# Exercise exr-emx-wv-drude-reflectivity: normal-incidence reflectivity of a Drude metal, omega in units of omega_p.
for w in [0.5, 0.99, 1.01, 1.1, 2]:
    eps = CDF(1 - 1 / (w * (w + 0.01 * I)))
    nn = sqrt(eps)
    if nn.imag() < 0:
        nn = -nn
    print('omega =', w, ': eps =', eps, ', R =', n(abs((1 - nn) / (1 + nn))^2, digits=3))
