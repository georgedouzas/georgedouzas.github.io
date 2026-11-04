# Exercise exr-emx-kk-lorentz: second Kramers-Kronig relation for eps = 1 + 1/(1 - w^2 - 0.2 i w).
def eps(x):
    return 1 + 1 / (1 - x^2 - 0.2 * I * x)
def reps1(x):
    return real_part(CDF(eps(x))) - 1
for w0 in [0.5, 2.0]:
    f = lambda x: (reps1(x) - reps1(w0)) / (x^2 - w0^2)
    val = -2 * w0 / pi * (numerical_integral(f, 0, w0)[0] + numerical_integral(f, w0, Infinity)[0])
    print('omega =', w0, ': integral =', n(val, digits=8), '; Im eps =', n(imag_part(CDF(eps(w0))), digits=8))
