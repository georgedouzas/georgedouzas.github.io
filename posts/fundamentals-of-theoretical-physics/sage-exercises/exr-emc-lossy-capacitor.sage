# Exercise exr-emc-lossy-capacitor: capacitor filled with Debye water, eps = 1 + 75/(1 - i omega tau0), tau0 = 8 ps.
C0 = 1e-12; V = 10; tau0 = 8e-12
for f in [50, 2.45e9]:
    w = 2 * pi * f
    eps = CDF(1 + 75 / (1 - I * w * tau0))
    Y = -I * w * eps * C0
    P = 0.5 * w * C0 * eps.imag() * V^2
    print(f, 'Hz: eps =', eps, '; admittance =', CDF(Y), 'S; <P> =', n(P, digits=3), 'W')
