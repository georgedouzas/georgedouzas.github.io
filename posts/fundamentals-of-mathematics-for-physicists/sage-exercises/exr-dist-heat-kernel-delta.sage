# Exercise exr-dist-heat-kernel-delta: the heat kernel K_t has unit integral, tends to delta as t -> 0+, and has second
# moment 2 kappa t.
x, t, kap = var("x t kappa")
assume(t > 0); assume(kap > 0)
K = exp(-x ** 2 / (4 * kap * t)) / sqrt(4 * pi * kap * t)
print("int K_t =", integrate(K, x, -oo, oo).canonicalize_radical())
for name, phi in [("cos x", cos(x)), ("x^2 + x + 1", x ** 2 + x + 1)]:
    v = integrate(K * phi, x, -oo, oo).canonicalize_radical()
    print("int K_t *", name, "=", v, "; limit t -> 0+:", limit(v, t=0, dir="+"))
print("second moment int K_t x^2 =", integrate(K * x ** 2, x, -oo, oo).canonicalize_radical())
