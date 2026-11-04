# Exercise exr-emx-sc-curve: sigma/sigma_T = w^4/((w0^2 - w^2)^2 + g^2 w^2) with w0 = 1, g = 0.05.
g = 0.05
for w in [0.1, 0.5, 1, 2, 10]:
    s = w^4 / ((1 - w^2)^2 + g^2 * w^2)
    print('omega =', w, ': sigma/sigma_T =', n(s, digits=3), '; Rayleigh omega^4 =', n(w^4, digits=3))
