# Exercise exr-qm-propagator-sage: the free propagator solves the Schroedinger equation; the Euclidean kernel tends to the identity.
x, y = var("x y", domain="real"); hbar, m, t, tau = var("hbar m t tau", domain="positive")
K = sqrt(m / (2 * pi * I * hbar * t)) * exp(I * m * (x - y) ** 2 / (2 * hbar * t))
print("(i hbar K_t + hbar^2/(2m) K_xx)/K =", ((I * hbar * diff(K, t) + hbar ** 2 / (2 * m) * diff(K, x, 2)) / K).simplify_full())
KE = sqrt(m / (2 * pi * hbar * tau)) * exp(-m * (x - y) ** 2 / (2 * hbar * tau))
r = integrate(KE * exp(-y ** 2), y, -oo, oo).canonicalize_radical()
print("int K_E(x, y; tau) e^(-y^2) dy =", r)
print("limit tau -> 0:", limit(r, tau=0))
