# Exercise exr-emx-en-gap: energy in the gap and in the iron of an electromagnet (SI).
mu0 = 4 * pi * 1e-7
N, I, mur, w, A = 400, 2, 1000, 1e-3, 10e-4
l = 2 * pi * 0.1 - w
B = mu0 * N * I / (w + l / mur)
Wg = B^2 / (2 * mu0) * A * w
Wi = B^2 / (2 * mu0 * mur) * A * l
print('B =', n(B, digits=3), 'T; gap:', n(Wg, digits=3), 'J; iron:', n(Wi, digits=3), 'J; ratio:', n(Wg / Wi, digits=3))
