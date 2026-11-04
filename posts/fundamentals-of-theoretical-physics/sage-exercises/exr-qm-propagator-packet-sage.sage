# Exercise exr-qm-propagator-packet-sage: the Euclidean kernel applied to a Gaussian, and continuation to real time.
x, y = var("x y", domain="real"); hbar, m, s, tau, t = var("hbar m s tau t", domain="positive")
KE = sqrt(m / (2 * pi * hbar * tau)) * exp(-m * (x - y) ** 2 / (2 * hbar * tau))
psi0 = (2 * pi * s ** 2) ** (-1 / 4) * exp(-y ** 2 / (4 * s ** 2))
out = integrate(KE * psi0, y, -oo, oo)
S2 = s ** 2 + hbar * tau / (2 * m)
claimed = (2 * pi * s ** 2) ** (-1 / 4) * s / sqrt(S2) * exp(-x ** 2 / (4 * S2))
print("result / claimed =", (out / claimed).canonicalize_radical())
print("sigma^2 + hbar tau/(2m) at tau = i t, divided by sigma^2 a(t):", (S2.subs(tau=I * t) / (s ** 2 * (1 + I * hbar * t / (2 * m * s ** 2)))).simplify_full())
