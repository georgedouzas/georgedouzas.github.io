# Exercise exr-emw-pw-laser: an intense laser pulse (SI units).
eps0, c = 8.854e-12, 2.998e8
I = 1.0 / 10e-15 / 10e-12
E0 = sqrt(2 * I / (c * eps0))
print('intensity %.2e W/m^2, field amplitude %.2e V/m, ratio to the atomic field 5.14e11 V/m: %.1f' % (I, E0, E0 / 5.14e11))
