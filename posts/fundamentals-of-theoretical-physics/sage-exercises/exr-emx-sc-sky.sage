# Exercise exr-emx-sc-sky: Rayleigh attenuation of sunlight in air (SI).
N = 2.5e25
for lam in [450, 700]:
    s = 5e-31 * (500 / lam)^4
    l = 1 / (N * s)
    print(lam, 'nm: mean free path =', n(l / 1e3, digits=3), 'km; transmission overhead =', n(exp(-8e3 / l), digits=3), '; at sunset =', n(exp(-38 * 8e3 / l), digits=3))
