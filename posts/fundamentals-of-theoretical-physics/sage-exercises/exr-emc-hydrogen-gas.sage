# Exercise exr-emc-hydrogen-gas: one-oscillator model of atomic hydrogen at 500 nm (SI).
e = 1.602176634e-19; me = 9.1093837e-31; eps0 = 8.8541878128e-12; c = 299792458; hbar = 1.054571817e-34
w0 = 10.2 * e / hbar
w = 2 * pi * c / 500e-9
N = 2.7e25
nm1 = N * e^2 / (2 * eps0 * me * (w0^2 - w^2))
re = e^2 / (4 * pi * eps0 * me * c^2)
sT = 8 * pi / 3 * re^2
s = sT * w^4 / (w0^2 - w^2)^2
print('omega0 =', n(w0, digits=3), '; omega =', n(w, digits=3))
print('n - 1 =', n(nm1, digits=3), '; sigma_T =', n(sT, digits=3), 'm^2; sigma =', n(s, digits=3), 'm^2')
