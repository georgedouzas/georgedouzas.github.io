# Exercise exr-qm-osc-ground-check: the two lowest oscillator states, with hbar, m, omega symbolic, and their normalisation.
x = var("x", domain="real"); hbar, m, w = var("hbar m w", domain="positive")
H = lambda f: -hbar ** 2 / (2 * m) * diff(f, x, 2) + m * w ** 2 * x ** 2 / 2 * f
for f, E in [(exp(-m * w * x ** 2 / (2 * hbar)), hbar * w / 2), (x * exp(-m * w * x ** 2 / (2 * hbar)), 3 * hbar * w / 2)]:
    N = 1 / sqrt(integrate(f ** 2, x, -oo, oo))
    print("H psi - E psi =", (H(f) - E * f).simplify_full(), "; E =", E, "; normalisation constant =", N.canonicalize_radical())
