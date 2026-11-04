# Exercise exr-emx-sc-angular: total Thomson cross-section from the angular distributions.
th, ph, re = var('theta phi r_e')
s1 = integrate(integrate(re^2 * sin(th)^2 * sin(th), th, 0, pi), ph, 0, 2 * pi)
s2 = integrate(integrate(re^2 * (1 + cos(th)^2) / 2 * sin(th), th, 0, pi), ph, 0, 2 * pi)
print('polarised:', s1, '; unpolarised:', s2, '; equal to 8 pi r_e^2/3:', bool(s1 == 8 * pi * re^2 / 3), bool(s2 == 8 * pi * re^2 / 3))
