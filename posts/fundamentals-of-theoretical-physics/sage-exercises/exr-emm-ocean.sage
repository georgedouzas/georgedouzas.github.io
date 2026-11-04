# Exercise exr-emm-ocean (Macchi et al., Problem 6.7): currents induced in sea water by the Earth's field (SI units).
sigma, v, B, rho_m = 4.0, 1.0, 5e-5, 1025.0
J = sigma * v * B; tau = rho_m / (sigma * B^2)
print('J = %.1e A/m^2; braking time rho/(sigma B^2) = %.2e s = %.0f years' % (J, tau, tau / 3.156e7))
