# Exercise exr-emw-dip-pulsar (Macchi et al., Problem 10.5): magnetic field of the Crab pulsar from its spin-down (SI units).
Ps, Pdot, Is, Rs = 0.033, 4.2e-13, 1e38, 1e4
mu0, c = 1.2566e-6, 2.998e8
w = 2 * pi.n() / Ps
Edot = 4 * pi.n()^2 * Is * Pdot / Ps^3
m = sqrt(6 * pi.n() * c^3 * Edot / (mu0 * w^4))
B = mu0 * m / (2 * pi.n() * Rs^3)
print('spin-down power %.2e W, dipole moment %.2e A m^2, polar field %.2e T' % (Edot, m, B))
