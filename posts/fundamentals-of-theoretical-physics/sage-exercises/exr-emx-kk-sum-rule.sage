# Exercise exr-emx-kk-sum-rule: f-sum rule for one Lorentz oscillator.
for (w0, g) in [(1, 0.1), (1, 1), (2, 0.5), (0.5, 3)]:
    f = lambda x: x * (g * x / ((w0^2 - x^2)^2 + g^2 * x^2))   # omega Im eps / omega_p^2
    val = numerical_integral(f, 0, Infinity)[0]
    print('omega0 =', w0, ', gamma =', g, ': integral/omega_p^2 =', n(val, digits=8), '; pi/2 =', n(pi / 2, digits=8))
