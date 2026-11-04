# Exercise exr-qm-propagator-composition-sage: the composition law of the Euclidean free kernel, with two and three kernels.
x, y, z, w = var("x y z w", domain="real"); hbar, m, t1, t2, t3 = var("hbar m t1 t2 t3", domain="positive")
KE = lambda X, Y, tt: sqrt(m / (2 * pi * hbar * tt)) * exp(-m * (X - Y) ** 2 / (2 * hbar * tt))
two = integrate(KE(x, z, t1) * KE(z, y, t2), z, -oo, oo)
print("two kernels / K_E(t1 + t2) =", (two / KE(x, y, t1 + t2)).canonicalize_radical())
three = integrate(integrate(KE(x, z, t1) * KE(z, w, t2), z, -oo, oo) * KE(w, y, t3), w, -oo, oo)
print("three kernels / K_E(t1 + t2 + t3) =", (three / KE(x, y, t1 + t2 + t3)).canonicalize_radical())
